"""
full_ai_1d.py  -  p(E | d) SENZA conoscenza fisica (nessun E^1.77, nessun k, nessun p fittato).

Modelli (scala dal meno al piu' strutturato):
  qnet            Q(tau, d) -> E, monotona in tau, loss sui quantili. Ipotesi nulla: nessun trasporto.
  auto_ode        dE/dz = -S(E, tau)   ODE autonoma in E (unica ipotesi: mezzo omogeneo in z)
  auto_ode_notau  dE/dz = -S(E)        come sopra ma S dipende solo da E (flusso deterministico)
  warp            u = g(E) con g monotona APPRESA (g(0)=0), la profondita' agisce come traslazione
                  u_tgt = u_src - 1*Delta_d  (la scala di c e' assorbita da g).  Se g impara da sola
                  qualcosa di simile a E^p, hai "riscoperto" la legge dai dati.

Protocollo (per non rendere il 3.5 cm un test non vergine):
  - default: solo fold leave-one-depth-out  (2.5<-2.0, 3.0<-2.5, 2.0<-1.0), il 3.5 NON viene caricato
  - 3.5 cm solo con --confirm_final; ogni uso viene registrato in eval/final_35_log.txt

Uso:
  python full_ai_1d.py --epochs 600 --seeds 1
  python full_ai_1d.py --methods warp,auto_ode --epochs 800
  python full_ai_1d.py --methods warp --confirm_final      # solo a configurazione congelata

Riferimento (stesso protocollo, stessa funzione di metrica): shift_k101 e fit_pk compaiono nella tabella.
"""
import argparse, csv, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ks_2samp, wasserstein_distance

from train_gen_energy_1d import (load_proton_energies, make_quantile_grid, fit_pk_scipy,
                                 physics_transport, interp1d_diff, mono, surviving_quantiles,
                                 quantile_loss, _mlp, _src_quantiles)

TRAIN6 = [0.5, 1.0, 2.0, 2.5, 3.0, 4.0]
FOLDS = [(2.5, 2.0), (3.0, 2.5), (2.0, 1.0)]   # (profondita' tenuta fuori, sorgente)
E_SCALE = 70.0
N_STEPS = 16
SP1 = 0.5413   # softplus(0.5413) = 1


# ------------------------------------------------------------------ utilita' condivise
def metrics(E_real, E_gen):
    std = E_real.std()
    w1 = wasserstein_distance(E_real, E_gen) / std
    qs = np.quantile(E_real, [0.01, 0.05])
    tr = E_real[E_real <= qs[1]]
    tg = E_gen[E_gen <= np.quantile(E_gen, 0.05)]
    ks = ks_2samp(tr / (tr.mean() + 1e-8), tg / (tg.mean() + 1e-8)).statistic
    return dict(w1=float(w1), ks=float(ks), dstd=float((E_gen.std() - std) / std),
                q1=float(100 * np.mean(E_gen <= qs[0])), q5=float(100 * np.mean(E_gen <= qs[1])),
                n=int(len(E_gen)))


def fmt(m):
    return (f"W1/std={m['w1']:.4f} KSt={m['ks']:.3f} dstd={m['dstd']:+.3f} "
            f"q1={m['q1']:.1f}% q5={m['q5']:.1f}% N={m['n']}")


def quantile_tables(E_by, depths, tau_np, device):
    q_np = {d: np.quantile(E_by[d], tau_np) for d in depths}
    q_t = {d: torch.from_numpy(q.astype(np.float32)).to(device) for d, q in q_np.items()}
    return q_np, q_t


def rank_sample(tau, curve, n):
    """Valuta la curva quantilica (resa monotona) ai ranghi di n campioni."""
    c = mono(curve)
    ranks = (np.arange(n, dtype=np.float32) + 0.5) / n
    u = torch.from_numpy(np.clip(ranks, 1e-4, 1 - 1e-4)).to(tau.device)
    return interp1d_diff(tau, c, u)


def train_pairs(model, q_dict, tau, epochs, lr, wd, max_dd=1.5, tag="", verbose=True):
    """Addestra un modello con model.predict(tau, q_src, d1, d2) -> (E_sopravvissuti_ai_livelli_tau, curva)."""
    depths = sorted(q_dict)
    pairs = [(a, b) for a in depths for b in depths if 0 < b - a <= max_dd + 1e-9]
    assert pairs, "nessuna coppia di profondita' per il training"
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        tot, mp = 0.0, 0.0
        for d1, d2 in pairs:
            E_pred, curve = model.predict(tau, q_dict[d1], d1, d2)
            mp = mp + torch.mean(F.relu(-(curve[1:] - curve[:-1])))
            tot = tot + quantile_loss(E_pred, q_dict[d2], tau)
        loss = tot / len(pairs) + 10.0 * mp / len(pairs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        if verbose and (ep == 1 or ep % max(1, epochs // 4) == 0):
            print(f"      [{tag}] ep {ep:4d}/{epochs} loss={loss.item():.4f} mono_pen={float(mp)/len(pairs):.1e}")
    return model


def summarize(rows, title):
    folds = sorted({r["fold"] for r in rows})
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    print(f"\n===== {title}: W1/std medio sui seed, per fold (KS coda tra parentesi) =====")
    print(f"{'metodo':18s}" + "".join(f"{('hold ' + str(f)):>16s}" for f in folds) + f"{'media LOO':>12s}")
    for m in methods:
        cells, w_loo = [], []
        for f in folds:
            sel = [r for r in rows if r["method"] == m and r["fold"] == f]
            w, k = np.mean([r["w1"] for r in sel]), np.mean([r["ks"] for r in sel])
            cells.append(f"{w:7.4f} ({k:.2f})")
            if f != 3.5:
                w_loo.append(w)
        print(f"{m:18s}" + "".join(f"{c:>16s}" for c in cells) + f"{np.mean(w_loo) if w_loo else float('nan'):12.4f}")


def save_rows(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# ------------------------------------------------------------------ modelli senza fisica
class QNetPure(nn.Module):
    """E(tau, d) in MeV, monotona in tau per costruzione (incrementi softplus). Nessun trasporto."""
    def __init__(self, hidden=128):
        super().__init__()
        self.inc = _mlp(2, hidden)
        nn.init.zeros_(self.inc[-1].weight); nn.init.constant_(self.inc[-1].bias, -7.6)
        self.start = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.start[-1].weight); nn.init.constant_(self.start[-1].bias, SP1)

    def curve(self, tau, d):
        tm = 0.5 * (tau[:-1] + tau[1:])
        feat = torch.stack([torch.logit(tm) / 6.0, torch.full_like(tm, d / 4.0)], dim=-1)
        dE = E_SCALE * F.softplus(self.inc(feat).squeeze(-1))
        e0 = 0.1 * F.softplus(self.start(torch.tensor([[d / 4.0]], device=tau.device)).squeeze())
        return torch.cat([e0.reshape(1), e0 + torch.cumsum(dE, 0)])

    @torch.no_grad()
    def sample(self, tau, d, n=500_000):
        return rank_sample(tau, self.curve(tau, d), n).clamp(min=1e-4).cpu().numpy()


def train_qnet(q_dict, tau, epochs, lr=3e-3, wd=1e-3):
    model = QNetPure().to(tau.device)
    depths = sorted(q_dict)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = sum(quantile_loss(model.curve(tau, d), q_dict[d], tau) for d in depths) / len(depths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        if ep == 1 or ep % max(1, epochs // 4) == 0:
            print(f"      [qnet] ep {ep:4d}/{epochs} loss={loss.item():.4f}")
    return model


class AutoODE(nn.Module):
    """dE/dz = -S(E[, tau]) in MeV/cm, autonoma in z. Protoni con E<=0 sono fermi (taglio in E a 0)."""
    def __init__(self, use_tau=True, hidden=96):
        super().__init__()
        self.use_tau = use_tau
        self.net = _mlp(2 if use_tau else 1, hidden)
        nn.init.normal_(self.net[-1].weight, std=0.01); nn.init.constant_(self.net[-1].bias, -1.9)

    def velocity(self, E, tau):
        x = (E / E_SCALE).clamp(min=0.0)
        inp = torch.stack([x, tau], dim=-1) if self.use_tau else x[:, None]
        return -F.softplus(self.net(inp).squeeze(-1)) * E_SCALE

    def integrate(self, tau, q_src, d1, d2):
        E = q_src
        dt = (d2 - d1) / N_STEPS
        for _ in range(N_STEPS):
            k1 = self.velocity(E, tau)
            k2 = self.velocity(E + 0.5 * dt * k1, tau)
            E = E + dt * k2
        return E

    def predict(self, tau, q_src, d1, d2):
        curve = self.integrate(tau, q_src, d1, d2)
        es, _ = surviving_quantiles(tau, curve, 0.0)
        return es.clamp(min=1e-6), curve

    @torch.no_grad()
    def sample_from_source(self, E_src, d1, d2, tau):
        curve = self.integrate(tau, _src_quantiles(E_src, tau), d1, d2)
        e = rank_sample(tau, curve, len(E_src))
        return e[e > 0].cpu().numpy()


class WarpTranslate(nn.Module):
    """u = g(E), g monotona appresa con g(0)=0; profondita' = traslazione u_tgt = u_src - Delta_d.
    L'inversa g^-1 e' interpolata da una tabella densa (E in [0, e_max])."""
    def __init__(self, hidden=64, n_tab=8192, e_max=80.0):
        super().__init__()
        self.net = _mlp(1, hidden)
        nn.init.zeros_(self.net[-1].weight); nn.init.constant_(self.net[-1].bias, -2.25)  # g'=0.1 -> ~10 MeV/cm
        self.register_buffer("E_tab", torch.linspace(0.0, e_max, n_tab))
        self.e_max = e_max

    def g_table(self):
        Em = 0.5 * (self.E_tab[1:] + self.E_tab[:-1])
        gp = F.softplus(self.net((Em / E_SCALE)[:, None]).squeeze(-1)) + 1e-4
        dE = self.E_tab[1:] - self.E_tab[:-1]
        return torch.cat([torch.zeros(1, device=Em.device), torch.cumsum(gp * dE, 0)])

    def predict(self, tau, q_src, d1, d2):
        g = self.g_table()
        u_t = interp1d_diff(self.E_tab, g, q_src.clamp(0.0, self.e_max)) - (d2 - d1)
        us, _ = surviving_quantiles(tau, u_t, 0.0)
        E = interp1d_diff(g, self.E_tab, us.clamp(min=0.0, max=float(g[-1])))
        return E.clamp(min=1e-6), u_t

    @torch.no_grad()
    def sample_from_source(self, E_src, d1, d2, tau):
        g = self.g_table()
        q = _src_quantiles(E_src, tau).clamp(0.0, self.e_max)
        u_t = interp1d_diff(self.E_tab, g, q) - (d2 - d1)
        us = rank_sample(tau, u_t, len(E_src))
        us = us[us > 0].clamp(max=float(g[-1]))
        return interp1d_diff(g, self.E_tab, us).cpu().numpy()

    @torch.no_grad()
    def exponent_report(self):
        """Pendenza log-log di g in alcuni punti: se g ~ E^p, la pendenza e' p."""
        g = self.g_table()
        out = []
        for e in (5.0, 10.0, 20.0, 40.0, 60.0):
            i = int(e / self.e_max * (len(g) - 1))
            j = max(1, int(0.9 * i)); k = min(len(g) - 1, int(1.1 * i))
            slope = torch.log(g[k] / g[j]) / torch.log(self.E_tab[k] / self.E_tab[j])
            out.append(f"E={e:.0f}: {float(slope):.2f}")
        return ", ".join(out)


# ------------------------------------------------------------------ fold
def run_fold(hold, src, E_by, a, tau, tau_np, dev, rows):
    train_depths = [d for d in TRAIN6 if d != hold]
    q_np, q_t = quantile_tables(E_by, train_depths, tau_np, dev)
    E_real, E_src = E_by[hold], E_by[src]
    print(f"\n######## FOLD: hold-out {hold} cm, sorgente {src} cm (Delta={hold - src:.2f}), "
          f"training su {train_depths} ########")

    p, k, _ = fit_pk_scipy(q_np, tau_np, max_depth=3.0)
    base = {"shift_k101": physics_transport(E_src, src, hold, 1.77, 450.0 * 1.01, 0.0),
            "fit_pk": physics_transport(E_src, src, hold, p, k, 0.0)}
    for name, E in base.items():
        m = metrics(E_real, E); rows.append(dict(fold=hold, method=name, seed=0, **m))
        print(f"  {name:16s}: {fmt(m)}")

    for meth in a.methods:
        for s in range(a.seeds):
            torch.manual_seed(42 + s)
            t0 = time.time()
            if meth == "qnet":
                model = train_qnet(q_t, tau, 2 * a.epochs)
                E_gen = model.sample(tau, hold)
            else:
                model = {"auto_ode": lambda: AutoODE(True), "auto_ode_notau": lambda: AutoODE(False),
                         "warp": lambda: WarpTranslate()}[meth]().to(dev)
                train_pairs(model, q_t, tau, a.epochs, lr=3e-3, wd=1e-3, max_dd=1.5, tag=meth)
                E_gen = model.sample_from_source(E_src, src, hold, tau)
                if meth == "warp":
                    print(f"      pendenza log-log di g(E) (p fisico ~1.77): {model.exponent_report()}")
            m = metrics(E_real, E_gen); rows.append(dict(fold=hold, method=meth, seed=s, **m))
            print(f"  {meth:16s} s{s}: {fmt(m)}  [{time.time() - t0:.0f}s]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/raw_phsp")
    ap.add_argument("--methods", default="qnet,auto_ode,auto_ode_notau,warp")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--confirm_final", action="store_true", help="valuta anche 3.5 cm (una volta, a config congelata)")
    ap.add_argument("--out", default="eval/full_ai_1d")
    a = ap.parse_args()
    a.methods = [m for m in a.methods.split(",") if m]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    depths = set(TRAIN6) | ({3.5} if a.confirm_final else set())
    print("=== DATI ===")
    E_by = load_proton_energies(a.data_dir, depths)
    tau = make_quantile_grid(2048, dev)
    tau_np = tau.cpu().numpy()

    rows = []
    for hold, src in FOLDS:
        run_fold(hold, src, E_by, a, tau, tau_np, dev, rows)
    if a.confirm_final:
        run_fold(3.5, 3.0, E_by, a, tau, tau_np, dev, rows)
        Path("eval").mkdir(exist_ok=True)
        with open("eval/final_35_log.txt", "a") as f:
            f.write(f"{time.ctime()} full_ai_1d methods={a.methods} epochs={a.epochs} seeds={a.seeds}\n")
        print("\n[!] Uso del 3.5 cm registrato in eval/final_35_log.txt")

    summarize(rows, "full AI 1D")
    save_rows(rows, f"{a.out}/results.csv")
    print(f"\nRisultati salvati in {a.out}/results.csv")


if __name__ == "__main__":
    main()
