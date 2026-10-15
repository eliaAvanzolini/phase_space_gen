"""
deep_ai_1d.py (v2 - con correzioni di Claude e validazione NIST PSTAR)
======================================================================
1. Peso uniforme sulle coppie (nessuna soppressione (1 - tau_cut)^2 su 3.0 -> 4.0 cm).
2. Identificabilita' tra g(E) e h: vincolo 1 + h > 0 e penalita' L2 su ||h||^2.
3. Consistenza di semigruppo: il trasporto d1 -> d2 deve coincidere con d1 -> d_mid -> d2.
4. Validazione esterna con la tabella ufficiale NIST PSTAR del range CSDA in acqua (2-70 MeV).
5. Valutazione su fold LOO + Bootstrap (3.5 cm toccato solo con --confirm_final).
"""
import argparse, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from full_ai_1d import (TRAIN6, FOLDS, metrics, fmt, quantile_tables, rank_sample, summarize, save_rows)
from train_gen_energy_1d import (load_proton_energies, make_quantile_grid, fit_pk_scipy,
                                 physics_transport, interp1d_diff, mono, surviving_quantiles,
                                 quantile_loss, _mlp, _src_quantiles)
from robust_1d import ResQ, fit_pk_window, train_pairs

# Tabella ufficiale NIST PSTAR / ICRU 49: CSDA Range di protoni in acqua liquida (g/cm^2 = cm)
NIST_PSTAR_E_MEV = np.array([2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=np.float64)
NIST_PSTAR_R_CM  = np.array([0.00772, 0.0363, 0.1230, 0.2540, 0.4260, 0.8853, 1.489, 2.227, 3.093, 4.080], dtype=np.float64)


class LogRangeNet(nn.Module):
    def __init__(self, hidden=48, n_tab=4096, e_min=1e-2, e_max=80.0, e_ref=45.0):
        super().__init__()
        self.e_max = float(e_max)
        self.s_ref = float(np.log(e_ref))
        s_tab = torch.linspace(float(np.log(e_min)), float(np.log(e_max)), n_tab)
        self.register_buffer("s_tab", s_tab)
        self.register_buffer("E_tab", torch.cat([torch.zeros(1), torch.exp(s_tab)]))

        self.elast_net = _mlp(1, hidden)
        nn.init.zeros_(self.elast_net[-1].weight)
        nn.init.constant_(self.elast_net[-1].bias, -0.37)  # Parte da esponente ~ 1.0 (retta)
        self.log_u_ref = nn.Parameter(torch.tensor(0.70, dtype=torch.float32))

    def g_table(self):
        sm = 0.5 * (self.s_tab[1:] + self.s_tab[:-1])
        ds = self.s_tab[1:] - self.s_tab[:-1]
        s_norm = ((sm - self.s_ref) / 2.0)[:, None]
        alpha = 0.5 + F.softplus(self.elast_net(s_norm).squeeze(-1))

        log_u_un = torch.cat([torch.zeros(1, device=sm.device), torch.cumsum(alpha * ds, dim=0)])
        ref_idx = int(torch.searchsorted(self.s_tab, torch.tensor(self.s_ref, device=sm.device)))
        log_u = self.log_u_ref + (log_u_un - log_u_un[ref_idx])
        u_tab = torch.cat([torch.zeros(1, device=sm.device), torch.exp(log_u)])
        return u_tab, alpha

    def forward_E_to_u(self, E, u_tab=None):
        if u_tab is None:
            u_tab, _ = self.g_table()
        return interp1d_diff(self.E_tab, u_tab, E.clamp(0.0, self.e_max))

    def inverse_u_to_E(self, u, u_tab=None):
        if u_tab is None:
            u_tab, _ = self.g_table()
        return interp1d_diff(u_tab, self.E_tab, u.clamp(0.0, float(u_tab[-1])))

    @torch.no_grad()
    def nist_pstar_report(self):
        """Confronta g_theta(E) appresa dai dati con il range CSDA reale NIST PSTAR."""
        u_tab, _ = self.g_table()
        e_t = torch.tensor(NIST_PSTAR_E_MEV, device=u_tab.device, dtype=torch.float32)
        g_val = self.forward_E_to_u(e_t, u_tab).cpu().numpy()

        # Fattore di proporzionalita' rispetto al range CSDA reale (ancorato a 60 MeV)
        scale = NIST_PSTAR_R_CM[-2] / (g_val[-2] + 1e-8)
        g_scaled = g_val * scale
        rel_err = np.mean(np.abs(g_scaled[1:] - NIST_PSTAR_R_CM[1:]) / NIST_PSTAR_R_CM[1:]) * 100.0
        r2 = 1.0 - np.sum((g_scaled - NIST_PSTAR_R_CM) ** 2) / np.sum((NIST_PSTAR_R_CM - NIST_PSTAR_R_CM.mean()) ** 2)

        slopes = []
        for e in (5.0, 10.0, 20.0, 40.0, 60.0):
            pair = torch.tensor([e * 0.95, e * 1.05], device=u_tab.device, dtype=torch.float32)
            up = self.forward_E_to_u(pair, u_tab)
            s = float(torch.log(up[1] / up[0]) / torch.log(pair[1] / pair[0]))
            slopes.append(f"E={e:.0f}:{s:.2f}")
        return f"R2_NIST={r2:.5f} (err_rel={rel_err:.2f}%) | pendenze [{', '.join(slopes)}] (NIST ~1.76-1.80)"


class DeepRangeResQ(nn.Module):
    """
    Rete 1: u = g_theta(E) (LogRangeNet)
    Rete 2: residuo vincolato 1 + h_phi > 0 con penalita' L2 su ||h_phi||^2 e consistenza di semigruppo.
    """
    def __init__(self, hidden_range=48, hidden_res=96):
        super().__init__()
        self.range_net = LogRangeNet(hidden=hidden_range)
        self.res_net = _mlp(3, hidden_res)
        nn.init.zeros_(self.res_net[-1].weight)
        nn.init.zeros_(self.res_net[-1].bias)

    def _eval_h(self, tau, d1, dd):
        inp = torch.stack([tau, torch.full_like(tau, d1 / 4.0), torch.full_like(tau, dd / 4.0)], dim=-1)
        raw = self.res_net(inp).squeeze(-1)
        # Vincolo 1 + h > 0: rate = softplus(raw + 0.5413) vale 1.0 quando raw = 0, ed e' sempre > 0
        rate = F.softplus(raw + 0.54132485)
        h = rate - 1.0
        return h

    def transport_u(self, tau, u_src, d1, d2, use_res=True):
        dd = float(d2 - d1)
        if not use_res:
            return u_src - dd, torch.zeros_like(tau)
        h = self._eval_h(tau, d1, dd)
        u_t = u_src - dd * (1.0 + h)
        return u_t, h

    def predict(self, tau, q_src, d1, d2, use_res=True):
        u_tab, alpha = self.range_net.g_table()
        u_src = self.range_net.forward_E_to_u(q_src, u_tab)
        u_t, h = self.transport_u(tau, u_src, d1, d2, use_res=use_res)
        us, _ = surviving_quantiles(tau, u_t, 0.0)
        E_pred = self.range_net.inverse_u_to_E(us, u_tab)

        # Consistenza di semigruppo: d1 -> d_mid -> d2 deve coincidere con d1 -> d2
        d_mid = 0.5 * (float(d1) + float(d2))
        u_mid, _ = self.transport_u(tau, u_src, d1, d_mid, use_res=use_res)
        u_two_step, _ = self.transport_u(tau, u_mid, d_mid, d2, use_res=use_res)
        semi_loss = torch.mean((u_t - u_two_step) ** 2)

        return E_pred.clamp(min=1e-6), u_t, h, alpha, semi_loss

    @torch.no_grad()
    def sample_from_source(self, E_src, d1, d2, tau):
        u_tab, _ = self.range_net.g_table()
        q = _src_quantiles(E_src, tau)
        u_src = self.range_net.forward_E_to_u(q, u_tab)
        u_t, _ = self.transport_u(tau, u_src, d1, d2, use_res=True)
        us = rank_sample(tau, u_t, len(E_src))
        us = us[us > 0]
        return self.range_net.inverse_u_to_E(us, u_tab).cpu().numpy()


def train_deep_resq(model, q_dict, tau, warmup=300, epochs=600, max_dd=1.5):
    depths = sorted(q_dict)
    pairs = [(a, b) for a in depths for b in depths if 0 < b - a <= max_dd + 1e-9]

    # Fase 1: g_theta(E) su tutte le coppie con peso uniforme 1.0 (nessuna soppressione di 3.0->4.0)
    opt1 = torch.optim.AdamW(model.range_net.parameters(), lr=4e-3, weight_decay=1e-3)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=warmup, eta_min=2e-4)
    for _ in range(1, warmup + 1):
        opt1.zero_grad()
        tot, sp = 0.0, 0.0
        for d1, d2 in pairs:
            E_pred, _, _, alpha, _ = model.predict(tau, q_dict[d1], d1, d2, use_res=False)
            tot = tot + quantile_loss(E_pred, q_dict[d2], tau)
            sp = torch.mean((alpha[1:] - alpha[:-1]) ** 2) * 50.0
        loss = tot / len(pairs) + sp
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.range_net.parameters(), 5.0)
        opt1.step()
        sched1.step()

    # Fase 2: Congiunta con penalita' di identificabilita' ||h||^2 e consistenza di semigruppo
    opt2 = torch.optim.AdamW([
        {"params": model.res_net.parameters(), "lr": 2e-3, "weight_decay": 5e-3},
        {"params": model.range_net.parameters(), "lr": 2e-4, "weight_decay": 1e-3},
    ])
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs, eta_min=1e-4)
    for _ in range(1, epochs + 1):
        opt2.zero_grad()
        tot, mp, h_pen, semi_tot, sp = 0.0, 0.0, 0.0, 0.0, 0.0
        for d1, d2 in pairs:
            E_pred, u_t, h, alpha, semi_l = model.predict(tau, q_dict[d1], d1, d2, use_res=True)
            tot = tot + quantile_loss(E_pred, q_dict[d2], tau)
            mp = mp + torch.mean(F.relu(-(u_t[1:] - u_t[:-1]) + 1e-4))
            h_pen = h_pen + torch.mean(h ** 2)
            semi_tot = semi_tot + semi_l
            sp = torch.mean((alpha[1:] - alpha[:-1]) ** 2) * 50.0

        n_p = len(pairs)
        loss = (tot + 10.0 * mp + 2.0 * h_pen + 5.0 * semi_tot) / n_p + sp
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt2.step()
        sched2.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/raw_phsp")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--boot", type=int, default=5)
    ap.add_argument("--confirm_final", action="store_true")
    ap.add_argument("--out", default="eval/deep_ai_1d")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    depths = set(TRAIN6) | ({3.5} if a.confirm_final else set())
    E_by = load_proton_energies(a.data_dir, depths)
    tau = make_quantile_grid(2048, dev)
    tau_np = tau.cpu().numpy()

    print("\n=== 1. CONFRONTO SU FOLD LOO (2.5, 3.0, 2.0 CM) E VERIFICA NIST PSTAR ===")
    rows = []
    for hold, src in FOLDS:
        train_d = [d for d in TRAIN6 if d != hold]
        q_np, q_t = quantile_tables(E_by, train_d, tau_np, dev)
        p, k = fit_pk_window(q_np, tau_np, None)

        # 1. shift_k101
        m_sh = metrics(E_by[hold], physics_transport(E_by[src], src, hold, 1.77, 450.0 * 1.01, 0.0))
        rows.append(dict(fold=hold, method="shift_k101", **m_sh))

        # 2. ResidualQ standard
        torch.manual_seed(42)
        rq = ResQ(p, k, 96, "full").to(dev)
        train_pairs(rq, q_t, tau, a.epochs, lr=2e-3, wd=5e-3, max_dd=1.5, tag="resq", verbose=False)
        m_rq = metrics(E_by[hold], rq.sample_from_source(E_by[src], src, hold, tau))
        rows.append(dict(fold=hold, method="ResidualQ", **m_rq))

        # 3. DeepRangeResQ (con semigruppo e vincolo 1+h>0)
        torch.manual_seed(42)
        drq = DeepRangeResQ().to(dev)
        train_deep_resq(drq, q_t, tau, warmup=a.warmup, epochs=a.epochs)
        m_drq = metrics(E_by[hold], drq.sample_from_source(E_by[src], src, hold, tau))
        rows.append(dict(fold=hold, method="DeepRangeResQ", **m_drq))

        print(f"\n--- Fold Hold-out {hold} cm (da {src} cm) ---")
        print(f"  shift_k101   : {fmt(m_sh)}")
        print(f"  ResidualQ    : {fmt(m_rq)}  (p={p:.3f}, k={k:.1f})")
        print(f"  DeepRangeResQ: {fmt(m_drq)}")
        print(f"  [NIST Check] : {drq.range_net.nist_pstar_report()}")

    summarize(rows, "LOO Folds: ResidualQ vs DeepRangeResQ")

    print(f"\n=== 2. BOOTSTRAP ({a.boot} RICAMPIONAMENTI): VERIFICA STABILITA' ===")
    rng = np.random.default_rng(0)
    b_rows = []
    for b in range(a.boot):
        Eb = {d: np.sort(rng.choice(E, size=len(E), replace=True)) for d, E in E_by.items()}
        for hold, src in FOLDS:
            train_d = [d for d in TRAIN6 if d != hold]
            q_np, q_t = quantile_tables(Eb, train_d, tau_np, dev)
            p, k = fit_pk_window(q_np, tau_np, None)

            torch.manual_seed(42 + b)
            rq = ResQ(p, k, 96, "full").to(dev)
            train_pairs(rq, q_t, tau, a.epochs, lr=2e-3, wd=5e-3, max_dd=1.5, tag="resq", verbose=False)
            m_rq = metrics(E_by[hold], rq.sample_from_source(Eb[src], src, hold, tau))
            b_rows.append(dict(boot=b, fold=hold, method="ResidualQ", **m_rq))

            torch.manual_seed(42 + b)
            drq = DeepRangeResQ().to(dev)
            train_deep_resq(drq, q_t, tau, warmup=a.warmup, epochs=a.epochs)
            m_drq = metrics(E_by[hold], drq.sample_from_source(Eb[src], src, hold, tau))
            b_rows.append(dict(boot=b, fold=hold, method="DeepRangeResQ", **m_drq))

    print(f"\n{'Metodo':16s}" + "".join(f"{('hold ' + str(h)):>22s}" for h, _ in FOLDS))
    for meth in ("ResidualQ", "DeepRangeResQ"):
        cells = []
        for hold, _ in FOLDS:
            w = [r["w1"] for r in b_rows if r["method"] == meth and r["fold"] == hold]
            ks = [r["ks"] for r in b_rows if r["method"] == meth and r["fold"] == hold]
            cells.append(f"{np.mean(w):.4f}+-{np.std(w):.4f} ({np.mean(ks):.2f})")
        print(f"{meth:16s}" + "".join(f"{c:>22s}" for c in cells))

        out_dir = Path(a.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        save_rows(rows, str(out_dir / "deep_v2_loo.csv"))

        if b_rows:
            save_rows(b_rows, str(out_dir / "deep_v2_bootstrap.csv"))

if __name__ == "__main__":
    main()
