"""
robust_1d.py  -  robustezza e approfondimento di ResidualQ e ODE (con p,k fittati).

Esperimenti (--exp):
  hp       sweep di iperparametri su fold leave-one-depth-out (2.5<-2.0, 3.0<-2.5, 2.0<-1.0).
           Si sceglie qui la configurazione, MAI sul 3.5.
  boot     bootstrap dei dati di training (ricampionamento dei protoni): variabilita' vera dovuta alla
           statistica finita, a differenza dei seed (ResidualQ parte da zero e varia pochissimo coi seed).
  density  diagnosi sulla densita' della griglia in profondita' (hold-out 3.0):
             A_sparse        train {0.5,1,1.5,2,4}            sorgente 2.0 (Delta 1.0)
             B_dense_src2.0  train {0.5,1,1.5,2,2.5,3.5,4}    sorgente 2.0 (Delta 1.0)  -> B-A = effetto densita'
             C_dense_src2.5  stesso train di B                 sorgente 2.5 (Delta 0.5)  -> C-B = effetto Delta
           NB: usa 3.5 e 1.5 come TRAINING: e' una diagnosi, non un numero da riportare per il test 3.5.
  final    test su 3.5 cm con configurazione congelata (richiede --confirm_final, registra l'uso).

Configurazione (default = quella usata finora):
  ResidualQ: --rq_hidden 96 --rq_wd 5e-3 --rq_maxdd 1.5 --rq_fitdd 0 --rq_feat full
  ODE:       --ode_hidden 96 --ode_wd 1e-3 --ode_maxdd 1.5 --ode_fitdd 0 --ode_z 1 --ode_tau 1
  (fitdd 0 = il fit di p,k usa tutte le coppie di profondita'; 0.5 / 1.0 = solo coppie con quel Delta massimo)

Esempi:
  python robust_1d.py --exp hp --grid small --epochs 600
  python robust_1d.py --exp boot --boot 5 --epochs 600
  python robust_1d.py --exp density
  python robust_1d.py --exp final --confirm_final --rq_wd 5e-4 --rq_fitdd 0.5
"""
import argparse, csv, itertools, time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
import torch
import torch.nn as nn
import torch.nn.functional as F

from full_ai_1d import (TRAIN6, FOLDS, metrics, fmt, quantile_tables, rank_sample, train_pairs, save_rows)
from train_gen_energy_1d import (load_proton_energies, make_quantile_grid, physics_transport,
                                 surviving_quantiles, _mlp, _src_quantiles, V_SCALE)

N_STEPS_ODE = 10


# ------------------------------------------------------------------ fit p,k con finestra di Delta
def fit_pk_window(q_np, tau_np, max_dd=None, max_depth=3.0):
    depths = sorted(d for d in q_np if d <= max_depth)
    mask = tau_np >= 0.05
    pairs = [(a, b) for a in depths for b in depths if b > a and (max_dd is None or b - a <= max_dd + 1e-9)]
    assert pairs, "nessuna coppia per il fit di p,k"
    best = None
    for p in np.linspace(1.65, 1.90, 26):
        def loss(k):
            L = 0.0
            for d1, d2 in pairs:
                v2 = np.maximum(q_np[d1][mask] ** p - k * (d2 - d1), 1e-4)
                L += np.mean(np.abs(v2 ** (1 / p) - q_np[d2][mask]))
            return L / len(pairs)
        r = minimize_scalar(loss, bounds=(300.0, 600.0), method="bounded")
        if best is None or r.fun < best[0]:
            best = (r.fun, float(p), float(r.x))
    return best[1], best[2]


# ------------------------------------------------------------------ modelli
class ResQ(nn.Module):
    """fit_pk + residuo appreso h(tau[, d_src], delta_d) * V_SCALE * delta_d. Parte esattamente da fit_pk."""
    def __init__(self, p, k, hidden=96, feat="full"):
        super().__init__()
        self.p, self.k, self.feat = float(p), float(k), feat
        self.net = _mlp(3 if feat == "full" else 2, hidden)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def _curve(self, tau, q_src, d1, dd):
        v = q_src.clamp(min=1e-3) ** self.p - self.k * dd
        cols = [tau]
        if self.feat == "full":
            cols.append(torch.full_like(tau, d1 / 4.0))
        cols.append(torch.full_like(tau, dd / 4.0))
        return v + self.net(torch.stack(cols, dim=-1)).squeeze(-1) * V_SCALE * dd

    def predict(self, tau, q_src, d1, d2):
        v = self._curve(tau, q_src, d1, d2 - d1)
        vs, _ = surviving_quantiles(tau, v, 0.0)
        return vs.clamp(min=1e-6) ** (1 / self.p), v

    @torch.no_grad()
    def sample_from_source(self, E_src, d1, d2, tau):
        v = self._curve(tau, _src_quantiles(E_src, tau), d1, d2 - d1)
        vs = rank_sample(tau, v, len(E_src))
        return (vs[vs > 0] ** (1 / self.p)).cpu().numpy()


class ODEv(nn.Module):
    """ODE in v=E^p (p fittato): dv/dz = -softplus(net(v[, tau][, z])). use_z=False -> campo autonomo."""
    def __init__(self, p, hidden=96, use_z=True, use_tau=True):
        super().__init__()
        self.p, self.use_z, self.use_tau = float(p), use_z, use_tau
        self.net = _mlp(1 + int(use_tau) + int(use_z), hidden)
        nn.init.normal_(self.net[-1].weight, std=0.01); nn.init.constant_(self.net[-1].bias, 0.2)

    def velocity(self, v, tau, z):
        f = [v]
        if self.use_tau:
            f.append(tau)
        if self.use_z:
            f.append(torch.full_like(tau, z / 4.0))
        return -F.softplus(self.net(torch.stack(f, dim=-1)).squeeze(-1))

    def integrate(self, tau, q_src, d1, d2):
        v = q_src.clamp(min=1e-3) ** self.p / V_SCALE
        dt = (d2 - d1) / N_STEPS_ODE
        z = float(d1)
        for _ in range(N_STEPS_ODE):
            k1 = self.velocity(v, tau, z)
            k2 = self.velocity(v + 0.5 * dt * k1, tau, z + 0.5 * dt)
            v = v + dt * k2
            z += dt
        return v * V_SCALE

    def predict(self, tau, q_src, d1, d2):
        v = self.integrate(tau, q_src, d1, d2)
        vs, _ = surviving_quantiles(tau, v, 0.0)
        return vs.clamp(min=1e-6) ** (1 / self.p), v

    @torch.no_grad()
    def sample_from_source(self, E_src, d1, d2, tau):
        v = self.integrate(tau, _src_quantiles(E_src, tau), d1, d2)
        vs = rank_sample(tau, v, len(E_src))
        return (vs[vs > 0] ** (1 / self.p)).cpu().numpy()


# ------------------------------------------------------------------ run di una configurazione
def evaluate_config(kind, cfg, E_by, train_depths, hold, src, E_real, tau, tau_np, dev, epochs, seed, verbose=False):
    q_np, q_t = quantile_tables(E_by, train_depths, tau_np, dev)
    p, k = fit_pk_window(q_np, tau_np, cfg.get("fit_dd"))
    torch.manual_seed(seed)
    if kind == "resq":
        model = ResQ(p, k, cfg["hidden"], cfg["feat"]).to(dev)
    else:
        model = ODEv(p, cfg["hidden"], cfg["use_z"], cfg["use_tau"]).to(dev)
    train_pairs(model, q_t, tau, epochs, lr=cfg["lr"], wd=cfg["wd"], max_dd=cfg["max_dd"], tag=kind, verbose=verbose)
    return metrics(E_real, model.sample_from_source(E_by[src], src, hold, tau)), (p, k)


def cfg_from_args(a):
    rq = dict(hidden=a.rq_hidden, wd=a.rq_wd, max_dd=a.rq_maxdd, fit_dd=(a.rq_fitdd or None), feat=a.rq_feat, lr=2e-3)
    ode = dict(hidden=a.ode_hidden, wd=a.ode_wd, max_dd=a.ode_maxdd, fit_dd=(a.ode_fitdd or None),
               use_z=bool(a.ode_z), use_tau=bool(a.ode_tau), lr=3e-3)
    return {"resq": rq, "ode": ode}


def cfg_str(c):
    return " ".join(f"{k}={v}" for k, v in c.items() if k != "lr")


# ------------------------------------------------------------------ esperimento: hp
def build_grid(kind, size):
    if kind == "resq":
        if size == "small":
            ax = dict(hidden=[96], wd=[0.0, 5e-3], max_dd=[0.5, 1.5], fit_dd=[0.5, None], feat=["full"])
        else:
            ax = dict(hidden=[32, 96], wd=[0.0, 5e-3], max_dd=[0.5, 1.0, 1.5], fit_dd=[0.5, 1.0, None],
                      feat=["full", "no_dsrc"])
        base = dict(lr=2e-3)
    else:
        if size == "small":
            ax = dict(hidden=[96], wd=[1e-3], max_dd=[1.5], fit_dd=[None],
                      zt=[(True, True), (False, True), (False, False)])
        else:
            ax = dict(hidden=[96], wd=[1e-3, 1e-2], max_dd=[0.5, 1.5], fit_dd=[None],
                      zt=[(True, True), (False, True), (False, False)])
        base = dict(lr=3e-3)
    keys = list(ax)
    out = []
    for vals in itertools.product(*[ax[k] for k in keys]):
        c = dict(zip(keys, vals)); c.update(base)
        if "zt" in c:
            c["use_z"], c["use_tau"] = c.pop("zt")
        out.append(c)
    return out


def exp_hp(a, E_by, tau, tau_np, dev):
    rows = []
    for kind in ("resq", "ode"):
        grid = build_grid(kind, a.grid)
        print(f"\n===== HP sweep {kind}: {len(grid)} configurazioni x {len(FOLDS)} fold =====")
        for i, cfg in enumerate(grid):
            per = []
            for hold, src in FOLDS:
                train_d = [d for d in TRAIN6 if d != hold]
                m, _ = evaluate_config(kind, cfg, E_by, train_d, hold, src, E_by[hold], tau, tau_np, dev,
                                       a.epochs, 42)
                per.append(m)
                rows.append(dict(kind=kind, cfg=cfg_str(cfg), fold=hold, **m))
            print(f"  [{i + 1:3d}/{len(grid)}] {cfg_str(cfg)}\n        W1 per fold: "
                  + " ".join(f"{m['w1']:.4f}" for m in per) + f" | media={np.mean([m['w1'] for m in per]):.4f}"
                  + f" | KS medio={np.mean([m['ks'] for m in per]):.3f}")
    print("\n===== RANKING (media W1 sui fold LOO) =====")
    for kind in ("resq", "ode"):
        agg = {}
        for r in rows:
            if r["kind"] == kind:
                agg.setdefault(r["cfg"], []).append((r["w1"], r["ks"]))
        ranked = sorted(agg.items(), key=lambda kv: np.mean([x[0] for x in kv[1]]))
        print(f"\n  -- {kind}: migliori 5 --")
        for c, v in ranked[:5]:
            print(f"   W1={np.mean([x[0] for x in v]):.4f} KS={np.mean([x[1] for x in v]):.3f}  {c}")
        print(f"  -- {kind}: peggiore --  W1={np.mean([x[0] for x in ranked[-1][1]]):.4f}  {ranked[-1][0]}")
    save_rows(rows, f"{a.out}/hp.csv")


# ------------------------------------------------------------------ esperimento: bootstrap
def exp_boot(a, E_by, tau, tau_np, dev):
    cfgs = cfg_from_args(a)
    rng = np.random.default_rng(0)
    rows = []
    for b in range(a.boot):
        Eb = {d: np.sort(rng.choice(E, size=len(E), replace=True)) for d, E in E_by.items()}
        for hold, src in FOLDS:
            train_d = [d for d in TRAIN6 if d != hold]
            for kind in ("resq", "ode"):
                m, (p, k) = evaluate_config(kind, cfgs[kind], Eb, train_d, hold, src, E_by[hold], tau, tau_np,
                                            dev, a.epochs, 42 + b)
                rows.append(dict(boot=b, fold=hold, method=kind, **m))
                print(f"  boot {b} hold {hold} {kind:5s} (p={p:.3f} k={k:.1f}): {fmt(m)}")
            m = metrics(E_by[hold], physics_transport(Eb[src], src, hold, 1.77, 450.0 * 1.01, 0.0))
            rows.append(dict(boot=b, fold=hold, method="shift_k101", **m))
    print(f"\n===== BOOTSTRAP ({a.boot} ricampionamenti): W1/std media +- std, KS coda media =====")
    print(f"{'metodo':12s}" + "".join(f"{('hold ' + str(h)):>22s}" for h, _ in FOLDS))
    for meth in ("shift_k101", "resq", "ode"):
        cells = []
        for hold, _ in FOLDS:
            w = [r["w1"] for r in rows if r["method"] == meth and r["fold"] == hold]
            k = [r["ks"] for r in rows if r["method"] == meth and r["fold"] == hold]
            cells.append(f"{np.mean(w):.4f}+-{np.std(w):.4f} ({np.mean(k):.2f})")
        print(f"{meth:12s}" + "".join(f"{c:>22s}" for c in cells))
    save_rows(rows, f"{a.out}/boot.csv")


# ------------------------------------------------------------------ esperimento: densita' griglia
def exp_density(a, E_by, tau, tau_np, dev):
    cfgs = cfg_from_args(a)
    hold = 3.0
    setups = {"A_sparse": ([0.5, 1.0, 1.5, 2.0, 4.0], 2.0),
              "B_dense_src2.0": ([0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0], 2.0),
              "C_dense_src2.5": ([0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0], 2.5)}
    rows = []
    print("\n[!] Diagnosi: questo test usa 3.5 e 1.5 come TRAINING. Non e' il test 3.5 della tesi.")
    for name, (train_d, src) in setups.items():
        m = metrics(E_by[hold], physics_transport(E_by[src], src, hold, 1.77, 450.0 * 1.01, 0.0))
        rows.append(dict(setup=name, method="shift_k101", **m))
        print(f"\n--- {name}: train {train_d}, sorgente {src} (Delta={hold - src:.1f}) ---")
        print(f"  shift_k101: {fmt(m)}")
        for kind in ("resq", "ode"):
            ms = []
            for s in range(a.seeds):
                m, _ = evaluate_config(kind, cfgs[kind], E_by, train_d, hold, src, E_by[hold], tau, tau_np, dev,
                                       a.epochs, 42 + s)
                ms.append(m)
                rows.append(dict(setup=name, method=kind, **m))
            print(f"  {kind:10s}: W1={np.mean([x['w1'] for x in ms]):.4f} KSt={np.mean([x['ks'] for x in ms]):.3f}")
    print("\nLettura: B-A = effetto della densita' del training a parita' di lunghezza del trasporto;"
          " C-B = effetto di accorciare il trasporto.")
    save_rows(rows, f"{a.out}/density.csv")


# ------------------------------------------------------------------ esperimento: final 3.5
def exp_final(a, E_by, tau, tau_np, dev):
    if not a.confirm_final:
        raise SystemExit("--exp final richiede --confirm_final (il 3.5 cm va guardato a configurazione congelata).")
    cfgs = cfg_from_args(a)
    hold, src = 3.5, 3.0
    q_np, _ = quantile_tables(E_by, TRAIN6, tau_np, dev)
    rows = []
    print("\n===== TEST FINALE 3.5 cm (config congelata) =====")
    for name, E in (("shift_k101", physics_transport(E_by[src], src, hold, 1.77, 450.0 * 1.01, 0.0)),):
        m = metrics(E_by[hold], E); rows.append(dict(method=name, seed=0, **m)); print(f"  {name:10s}: {fmt(m)}")
    for kind in ("resq", "ode"):
        for s in range(a.seeds):
            m, (p, k) = evaluate_config(kind, cfgs[kind], E_by, TRAIN6, hold, src, E_by[hold], tau, tau_np, dev,
                                        a.epochs, 42 + s)
            rows.append(dict(method=kind, seed=s, **m))
            print(f"  {kind:10s} s{s} (p={p:.3f} k={k:.1f}): {fmt(m)}")
    Path("eval").mkdir(exist_ok=True)
    with open("eval/final_35_log.txt", "a") as f:
        f.write(f"{time.ctime()} robust_1d final cfg={cfgs} epochs={a.epochs} seeds={a.seeds}\n")
    print("\n[!] Uso del 3.5 cm registrato in eval/final_35_log.txt")
    save_rows(rows, f"{a.out}/final.csv")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/raw_phsp")
    ap.add_argument("--exp", choices=["hp", "boot", "density", "final"], required=True)
    ap.add_argument("--grid", choices=["small", "full"], default="small")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--boot", type=int, default=5)
    ap.add_argument("--confirm_final", action="store_true")
    ap.add_argument("--out", default="eval/robust_1d")
    ap.add_argument("--rq_hidden", type=int, default=96)
    ap.add_argument("--rq_wd", type=float, default=5e-3)
    ap.add_argument("--rq_maxdd", type=float, default=1.5)
    ap.add_argument("--rq_fitdd", type=float, default=0.0)
    ap.add_argument("--rq_feat", choices=["full", "no_dsrc"], default="full")
    ap.add_argument("--ode_hidden", type=int, default=96)
    ap.add_argument("--ode_wd", type=float, default=1e-3)
    ap.add_argument("--ode_maxdd", type=float, default=1.5)
    ap.add_argument("--ode_fitdd", type=float, default=0.0)
    ap.add_argument("--ode_z", type=int, default=1)
    ap.add_argument("--ode_tau", type=int, default=1)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    depths = set(TRAIN6)
    if a.exp == "density":
        depths |= {1.5, 3.5}
    if a.exp == "final":
        depths |= {3.5}
    print("=== DATI ===")
    E_by = load_proton_energies(a.data_dir, depths)
    tau = make_quantile_grid(2048, dev)
    tau_np = tau.cpu().numpy()
    Path(a.out).mkdir(parents=True, exist_ok=True)

    {"hp": exp_hp, "boot": exp_boot, "density": exp_density, "final": exp_final}[a.exp](a, E_by, tau, tau_np, dev)


if __name__ == "__main__":
    main()
