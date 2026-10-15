"""Ablation LOO per DeepRangeResQ.

Confronta, sugli stessi fold e con le stesse quantile-grid:
  1) shift_k101: baseline fisica;
  2) range_only: g(E) appresa, h=0;
  3) fixed_power_residual: g(E)=E**p/k fissata, h appresa;
  4) deep_full: g ed h entrambe apprese.

Dipende dai moduli gia' presenti nel progetto: deep_ai_1d.py,
full_ai_1d.py e train_gen_energy_1d.py.
Non usa il fold finale 3.5 cm: la valutazione e' LOO sui fold definiti in FOLDS.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_ai_1d import DeepRangeResQ, train_deep_resq
from full_ai_1d import TRAIN6, FOLDS, metrics, fmt, quantile_tables, rank_sample, save_rows
from train_gen_energy_1d import (
    load_proton_energies,
    make_quantile_grid,
    physics_transport,
    interp1d_diff,
    quantile_loss,
    _src_quantiles,
)


class FixedPowerRangeNet(nn.Module):
    """Range fissato g(E)=E**p/k, espresso in cm, con interfaccia LogRangeNet."""
    def __init__(self, p: float = 1.77, k: float = 450.0,
                 n_tab: int = 4096, e_min: float = 1e-2, e_max: float = 80.0):
        super().__init__()
        if p <= 0 or k <= 0:
            raise ValueError("p e k devono essere positivi")
        self.p = float(p)
        self.k = float(k)
        self.e_max = float(e_max)
        s_tab = torch.linspace(float(np.log(e_min)), float(np.log(e_max)), n_tab)
        e_tab = torch.cat([torch.zeros(1, dtype=s_tab.dtype), torch.exp(s_tab)])
        u_tab = e_tab.pow(self.p) / self.k
        self.register_buffer("E_tab", e_tab)
        self.register_buffer("u_fixed", u_tab)

    def g_table(self):
        alpha = torch.full(
            (len(self.E_tab) - 1,), self.p,
            dtype=self.E_tab.dtype, device=self.E_tab.device
        )
        return self.u_fixed, alpha

    def forward_E_to_u(self, E, u_tab=None):
        if u_tab is None:
            u_tab = self.u_fixed
        return interp1d_diff(self.E_tab, u_tab, E.clamp(0.0, self.e_max))

    def inverse_u_to_E(self, u, u_tab=None):
        if u_tab is None:
            u_tab = self.u_fixed
        return interp1d_diff(u_tab, self.E_tab, u.clamp(0.0, float(u_tab[-1])))



def sample_from_source(model, E_src, d1, d2, tau, use_res: bool = True):
    """Stessa procedura di campionamento del modello, con possibilita' h=0."""
    with torch.no_grad():
        u_tab, _ = model.range_net.g_table()
        q_src = _src_quantiles(E_src, tau)
        u_src = model.range_net.forward_E_to_u(q_src, u_tab)
        u_t, _ = model.transport_u(tau, u_src, d1, d2, use_res=use_res)
        sampled_u = rank_sample(tau, u_t, len(E_src))
        sampled_u = sampled_u[sampled_u > 0]
        out = model.range_net.inverse_u_to_E(sampled_u, u_tab)
        return out.detach().cpu().numpy()


def _pairs_from_q(q_dict, max_dd=1.5):
    depths = sorted(q_dict)
    pairs = [(d1, d2) for d1 in depths for d2 in depths
             if 0 < d2 - d1 <= max_dd + 1e-9]
    if not pairs:
        raise RuntimeError("Nessuna coppia di profondita' valida per l'addestramento")
    return pairs


def train_range_only(model, q_dict, tau, warmup=300, max_dd=1.5):
    """Ablation A: addestra g nel warm-up; h non viene usata ne' addestrata."""
    pairs = _pairs_from_q(q_dict, max_dd=max_dd)
    params = list(model.range_net.parameters())
    if not params:
        raise ValueError("range_only richiede una LogRangeNet apprendibile")
    opt = torch.optim.AdamW(params, lr=4e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, warmup), eta_min=2e-4)
    model.train()
    for ep in range(1, warmup + 1):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        smooth = 0.0
        for d1, d2 in pairs:
            pred, _, _, alpha, _ = model.predict(tau, q_dict[d1], d1, d2, use_res=False)
            total = total + quantile_loss(pred, q_dict[d2], tau)
            smooth = smooth + 50.0 * torch.mean((alpha[1:] - alpha[:-1]) ** 2)
        loss = total / len(pairs) + smooth / len(pairs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.range_net.parameters(), 5.0)
        opt.step()
        sched.step()
    return model


def train_residual_only(model, q_dict, tau, epochs=600, max_dd=1.5):
    """Ablation B: g(E) fissa; addestra h con penalita' originali su h/monotonia/semigruppo."""
    pairs = _pairs_from_q(q_dict, max_dd=max_dd)
    params = list(model.res_net.parameters())
    if not params:
        raise ValueError("res_net non contiene parametri apprendibili")
    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=1e-4)
    model.train()
    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        mono_pen = 0.0
        h_pen = 0.0
        semi_total = 0.0
        for d1, d2 in pairs:
            pred, u_t, h, _alpha, semi_loss = model.predict(tau, q_dict[d1], d1, d2, use_res=True)
            total = total + quantile_loss(pred, q_dict[d2], tau)
            mono_pen = mono_pen + torch.mean(F.relu(-(u_t[1:] - u_t[:-1]) + 1e-4))
            h_pen = h_pen + torch.mean(h ** 2)
            semi_total = semi_total + semi_loss
        n = len(pairs)
        loss = (total + 10.0 * mono_pen + 2.0 * h_pen + 5.0 * semi_total) / n
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.res_net.parameters(), 5.0)
        opt.step()
        sched.step()
    return model


def evaluate_one(model, E_by, src, hold, tau, use_res=True):
    samples = sample_from_source(model, E_by[src], src, hold, tau, use_res=use_res)
    return metrics(E_by[hold], samples), len(samples)


def main():
    ap = argparse.ArgumentParser(description="Ablation LOO DeepRangeResQ")
    ap.add_argument("--data_dir", default="data/raw_phsp")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only_folds", type=float, nargs="+", default=None,
                    help="fold LOO da eseguire, es. --only_folds 3.0; default: tutti i FOLDS")
    ap.add_argument("--fixed_p", type=float, default=1.77)
    ap.add_argument("--fixed_k", type=float, default=450.0,
                    help="g(E)=E**p/k; la coordinata u e' espressa in cm")
    ap.add_argument("--out", default="eval/deep_ai_1d_ablation")
    ap.add_argument("--methods", nargs="+",
                    choices=["shift", "range_only", "fixed_power_residual", "deep_full"],
                    default=["shift", "range_only", "fixed_power_residual", "deep_full"])
    a = ap.parse_args()

    if a.warmup < 1 or a.epochs < 1:
        raise ValueError("--warmup e --epochs devono essere >= 1")

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)

    depths = set(TRAIN6)
    E_by = load_proton_energies(a.data_dir, depths)
    tau = make_quantile_grid(2048, device)
    tau_np = tau.detach().cpu().numpy()

    rows = []
    folds = list(FOLDS)
    if a.only_folds:
        folds = [(hold, src) for hold, src in folds
                 if any(np.isclose(float(hold), wanted) for wanted in a.only_folds)]
        if not folds:
            raise ValueError(f"Nessun fold in FOLDS corrisponde a {a.only_folds}; fold disponibili: {[h for h, _ in FOLDS]}")
    print(f"Device={device}; train depths={sorted(TRAIN6)}; folds={[h for h, _ in folds]}; methods={a.methods}", flush=True)
    for fold_i, (hold, src) in enumerate(folds):
        train_d = [d for d in TRAIN6 if d != hold]
        _q_np, q_t = quantile_tables(E_by, train_d, tau_np, device)
        print(f"\n=== Fold hold={hold} cm, source={src} cm ===", flush=True)

        if "shift" in a.methods:
            pred = physics_transport(E_by[src], src, hold, 1.77, 450.0 * 1.01, 0.0)
            m = metrics(E_by[hold], pred)
            row = dict(fold=hold, source=src, method="shift_k101", n_generated=len(pred), **m)
            rows.append(row)
            print(f"shift_k101             {fmt(m)}", flush=True)

        if "range_only" in a.methods:
            torch.manual_seed(a.seed)
            np.random.seed(a.seed)
            model = DeepRangeResQ().to(device)
            train_range_only(model, q_t, tau, warmup=a.warmup)
            m, n_gen = evaluate_one(model, E_by, src, hold, tau, use_res=False)
            row = dict(fold=hold, source=src, method="range_only_g_learned", n_generated=n_gen, **m)
            rows.append(row)
            print(f"range_only (g, h=0)    {fmt(m)}; n={n_gen}", flush=True)

        if "fixed_power_residual" in a.methods:
            torch.manual_seed(a.seed)
            np.random.seed(a.seed)
            model = DeepRangeResQ().to(device)
            model.range_net = FixedPowerRangeNet(p=a.fixed_p, k=a.fixed_k).to(device)
            train_residual_only(model, q_t, tau, epochs=a.epochs)
            m, n_gen = evaluate_one(model, E_by, src, hold, tau, use_res=True)
            row = dict(fold=hold, source=src, method="fixed_power_plus_h", n_generated=n_gen, **m)
            rows.append(row)
            print(f"fixed power + h        {fmt(m)}; n={n_gen}", flush=True)

        if "deep_full" in a.methods:
            torch.manual_seed(a.seed)
            np.random.seed(a.seed)
            model = DeepRangeResQ().to(device)
            train_deep_resq(model, q_t, tau, warmup=a.warmup, epochs=a.epochs)
            m, n_gen = evaluate_one(model, E_by, src, hold, tau, use_res=True)
            row = dict(fold=hold, source=src, method="deep_range_resq_full", n_generated=n_gen, **m)
            rows.append(row)
            print(f"DeepRangeResQ completo {fmt(m)}; n={n_gen}", flush=True)
            try:
                print("NIST check: " + model.range_net.nist_pstar_report(), flush=True)
            except Exception as exc:
                print(f"NIST check non disponibile: {exc}", flush=True)

    csv_path = outdir / "deep_ai_1d_ablation_loo.csv"
    save_rows(rows, str(csv_path))
    print(f"\nRisultati salvati: {csv_path}", flush=True)
    print("\nRiepilogo medio per metodo:")
    for meth in sorted({r["method"] for r in rows}):
        group = [r for r in rows if r["method"] == meth]
        keys = [k for k in ("w1", "ks", "dstd") if all(k in r for r in group)]
        vals = ", ".join(f"{k}={np.mean([r[k] for r in group]):.5f}" for k in keys)
        print(f"  {meth:24s} n_folds={len(group)} {vals}")


if __name__ == "__main__":
    main()
