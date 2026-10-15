"""
train_gen_energy_1d.py  (versione corretta)
===========================================
Confronto 1D su p(E | d) con test hold-out a 3.5 cm, poi (opzionale) phase space 7D con cfm_condE.

Metodi:
  1. shift_k101  : baseline fisica manuale (p=1.77, k=450*1.01)
  2. fit_pk      : baseline fisica con p,k fittati (solo profondita' <= 3.0)
  3. PureNeuralODE: campo di velocita' in v=E^p appreso (nessun k fissato)
  4. ResidualQ   : fit_pk + residuo appreso h(tau, d_src, delta_d)

Correzioni rispetto alla versione precedente:
  - taglio di arresto fisico v > 0 (--v_cut, default 0), non E_min_osservata^p
  - monotonia della curva imposta con cummax in tutto il codice (taglio, interpolazione, campionamento)
  - n_steps ODE identico in training e generazione
  - piu' seed per i modelli appresi (media / min / max)
  - stampa penalita' di monotonia, frazione di sopravvissuti, confronto taglio v=0 vs E_min
  - salva le energie generate in .npy per eval successivi
  - il confronto dichiara quali profondita' ha visto ogni metodo

Uso rapido:  python train_gen_energy_1d.py --only_1d
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import ks_2samp, wasserstein_distance
import torch
import torch.nn as nn
import torch.nn.functional as F

import read_topas_phsp as rtp

TRAIN_DEPTHS = {0.5, 1.0, 2.0, 2.5, 3.0, 4.0}
V_SCALE = 500.0
N_STEPS = 10  # passi ODE, uguali in training e generazione


# ------------------------------------------------------------------ dati
def _energies_from_phsp(path):
    df = rtp.read_phsp(str(path))
    if "pdg" in df.columns:
        df = df[df["pdg"] == 2212]
    E = rtp.to_7d(df)[:, 6].astype(np.float64)
    return np.sort(E[E > 0.0])


def load_proton_energies(data_dir, allowed=TRAIN_DEPTHS):
    out = {}
    for p in sorted(Path(data_dir).glob("*.phsp")):
        if "peak" not in p.name.lower():
            continue
        d = rtp.parse_depth_from_name(p.name)
        if d is None or round(float(d), 3) not in allowed:
            continue
        E = _energies_from_phsp(p)
        out[round(float(d), 3)] = E
        print(f"  [Data] {p.name:32s} d={d:4.2f} | N={len(E):8d} | E_mean={E.mean():6.2f} | E_min={E.min():5.2f}")
    return out


def make_quantile_grid(n=2048, device="cpu"):
    u = np.linspace(0.0, 1.0, n, dtype=np.float64)
    tau = 0.5 + 0.5 * np.tanh(2.2 * (u - 0.5)) / np.tanh(1.1)
    tau = np.clip(tau, 1e-4, 1 - 1e-4).astype(np.float32)
    tau[0], tau[-1] = 1e-4, 1 - 1e-4
    return torch.from_numpy(tau).to(device)


# ------------------------------------------------------------------ metriche
def eval_1d_metrics(E_real, E_gen, label):
    std_r = E_real.std()
    w1 = wasserstein_distance(E_real, E_gen) / std_r
    dm = (E_gen.mean() - E_real.mean()) / std_r
    ds = (E_gen.std() - std_r) / std_r
    tr = E_real[E_real <= np.quantile(E_real, 0.05)]
    tg = E_gen[E_gen <= np.quantile(E_gen, 0.05)]
    ks = ks_2samp(tr / (tr.mean() + 1e-8), tg / (tg.mean() + 1e-8)).statistic
    qs = np.quantile(E_real, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95])
    m = [100.0 * np.mean(E_gen <= q) for q in qs]
    print(f"  {label:26s} | W1/std={w1:.4f} dmean={dm:+.4f} dstd={ds:+.4f} KSt={ks:.3f} | "
          f"q1={m[0]:4.1f} q5={m[1]:4.1f} q25={m[2]:4.1f} q50={m[3]:4.1f} q75={m[4]:4.1f} q95={m[5]:4.1f} "
          f"| N={len(E_gen)}")
    return w1


# ------------------------------------------------------------------ fisica
def fit_pk_scipy(q_np, tau_np, max_depth=3.0):
    depths = sorted(d for d in q_np if d <= max_depth)
    mask = tau_np >= 0.05
    best = None
    for p in np.linspace(1.65, 1.90, 51):
        def loss(k):
            L = 0.0
            for d1 in depths:
                for d2 in depths:
                    if d2 > d1:
                        v2 = np.maximum(q_np[d1][mask] ** p - k * (d2 - d1), 1e-4)
                        L += np.mean(np.abs(v2 ** (1 / p) - q_np[d2][mask]))
            return L
        r = minimize_scalar(loss, bounds=(300.0, 600.0), method="bounded")
        if best is None or r.fun < best[0]:
            best = (r.fun, float(p), float(r.x))
    return best[1], best[2], best[0]


def physics_transport(E_src, d_src, d_tgt, p, k, v_cut=0.0):
    v = E_src ** p - k * (d_tgt - d_src)
    return v[v > v_cut] ** (1.0 / p)


# ------------------------------------------------------------------ utilita' torch
def interp1d_diff(x, y, xq):
    idx = torch.searchsorted(x, xq).clamp(1, len(x) - 1)
    x0, x1, y0, y1 = x[idx - 1], x[idx], y[idx - 1], y[idx]
    w = ((xq - x0) / (x1 - x0 + 1e-8)).clamp(0.0, 1.0)
    return y0 + w * (y1 - y0)


def mono(v):
    return torch.cummax(v, dim=0)[0]


def surviving_quantiles(tau_grid, v_curve, v_cut):
    """Quantili (agli stessi livelli tau_grid) dei soli protoni con v > v_cut. Usa sempre la curva monotona."""
    v = mono(v_curve)
    if float(v[0]) >= v_cut:
        tau_cut = torch.zeros((), device=v.device, dtype=v.dtype)
    else:
        idx = torch.searchsorted(v.detach(), torch.tensor(v_cut, device=v.device, dtype=v.dtype)).clamp(1, len(v) - 1)
        v0, v1, t0, t1 = v[idx - 1], v[idx], tau_grid[idx - 1], tau_grid[idx]
        frac = ((v_cut - v0) / (v1 - v0 + 1e-6)).clamp(0.0, 1.0)
        tau_cut = (t0 + frac * (t1 - t0)).clamp(0.0, 0.96)
    tau_s = tau_cut + tau_grid * (1.0 - tau_cut)
    return interp1d_diff(tau_grid, v, tau_s), tau_cut


def quantile_loss(E_pred, E_tgt, tau_grid):
    l1 = torch.mean(torch.abs(E_pred - E_tgt))
    w = torch.exp(-tau_grid / 0.10)
    rel = torch.mean(w * torch.abs(E_pred - E_tgt) / (E_tgt + 1.0))
    return l1 + 2.5 * rel


def _sample_via_ranks(tau_grid, v_grid, n_src, p, v_cut):
    v_grid = mono(v_grid)
    ranks = (np.arange(n_src, dtype=np.float32) + 0.5) / n_src
    u = torch.from_numpy(np.clip(ranks, 1e-4, 1 - 1e-4)).to(tau_grid.device)
    v = interp1d_diff(tau_grid, v_grid, u)
    v = v[v > v_cut]
    return torch.pow(v, 1.0 / p).cpu().numpy()


def _src_quantiles(E_src, tau_grid):
    return torch.from_numpy(np.quantile(E_src, tau_grid.cpu().numpy()).astype(np.float32)).to(tau_grid.device)


def _mlp(n_in, hidden):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))


# ------------------------------------------------------------------ modelli
class PureNeuralODEFlow1D(nn.Module):
    def __init__(self, p_coord, hidden=96):
        super().__init__()
        self.p = float(p_coord)
        self.net = _mlp(3, hidden)
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.constant_(self.net[-1].bias, 0.2)

    def velocity(self, v, tau, z):
        inp = torch.stack([v, tau, torch.full_like(tau, z / 4.0)], dim=-1)
        return -F.softplus(self.net(inp).squeeze(-1))

    def integrate(self, tau, E_src_q, d_src, d_tgt):
        v = torch.pow(E_src_q.clamp(min=1e-3), self.p) / V_SCALE
        dt = (d_tgt - d_src) / N_STEPS
        z = float(d_src)
        for _ in range(N_STEPS):
            k1 = self.velocity(v, tau, z)
            k2 = self.velocity(v + 0.5 * dt * k1, tau, z + 0.5 * dt)
            v = v + dt * k2
            z += dt
        return v * V_SCALE

    @torch.no_grad()
    def sample_from_source(self, E_src, d_src, d_tgt, tau, v_cut):
        vg = self.integrate(tau, _src_quantiles(E_src, tau), d_src, d_tgt)
        return _sample_via_ranks(tau, vg, len(E_src), self.p, v_cut)


class LearnedQuantileResidual1D(nn.Module):
    def __init__(self, p_fit, k_fit, hidden=96):
        super().__init__()
        self.p, self.k = float(p_fit), float(k_fit)
        self.net = _mlp(3, hidden)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def transport(self, tau, E_src_q, d_src, d_tgt):
        dd = float(d_tgt - d_src)
        v = torch.pow(E_src_q.clamp(min=1e-3), self.p) - self.k * dd
        inp = torch.stack([tau, torch.full_like(tau, d_src / 4.0), torch.full_like(tau, dd / 4.0)], dim=-1)
        return v + self.net(inp).squeeze(-1) * V_SCALE * dd

    @torch.no_grad()
    def sample_from_source(self, E_src, d_src, d_tgt, tau, v_cut):
        vg = self.transport(tau, _src_quantiles(E_src, tau), d_src, d_tgt)
        return _sample_via_ranks(tau, vg, len(E_src), self.p, v_cut)


# ------------------------------------------------------------------ training
def _pairs(depths):
    return [(a, b) for a in depths for b in depths if 0 < (b - a) <= 1.5]


def _train(model, fwd, q_dict, tau, epochs, lr, wd, v_cut, tag):
    depths = sorted(q_dict)
    pairs = _pairs(depths)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        tot, mp = 0.0, 0.0
        for d1, d2 in pairs:
            v_t = fwd(tau, q_dict[d1], d1, d2)
            mp = mp + torch.mean(F.relu(-(v_t[1:] - v_t[:-1]) + 1e-3))
            vs, _ = surviving_quantiles(tau, v_t, v_cut)
            E_pred = torch.pow(vs.clamp(min=1e-6), 1.0 / model.p)
            tot = tot + quantile_loss(E_pred, q_dict[d2], tau)
        loss = tot / len(pairs) + 10.0 * mp
        loss.backward()
        opt.step()
        sched.step()
        if ep % 300 == 0 or ep == 1:
            print(f"    [{tag}] ep {ep:4d}/{epochs} loss={loss.item():.4f} mono_pen={float(mp)/len(pairs):.2e}")
    return model


def train_ode(q_dict, tau, p_coord, epochs, v_cut, seed, tag="ODE"):
    torch.manual_seed(seed)
    m = PureNeuralODEFlow1D(p_coord).to(tau.device)
    return _train(m, m.integrate, q_dict, tau, epochs, 3e-3, 1e-3, v_cut, tag)


def train_res(q_dict, tau, p_fit, k_fit, epochs, v_cut, seed, tag="ResQ"):
    torch.manual_seed(seed)
    m = LearnedQuantileResidual1D(p_fit, k_fit).to(tau.device)
    return _train(m, m.transport, q_dict, tau, epochs, 2e-3, 5e-3, v_cut, tag)


# ------------------------------------------------------------------ 7D (identico a generate_depth_condE)
@torch.no_grad()
def generate_full_7d_with_condE(condE_run, depth, E_phys, out_h5, steps=200, chunk=500_000, seed=0):
    from generate_depth_condE import load_condE_model
    from phsp_transforms import norm_depth, stats_at, e_fwd, Z0_CM
    model, stats, dev = load_condE_model(condE_run, None)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n = len(E_phys)
    mu, sd = stats_at(stats, float(depth))
    fE = e_fwd(E_phys, stats.get("etrans", "lin"), stats.get("p", 1.77))
    E_norm = ((fE - mu[3]) / sd[3]).astype(np.float32)
    d_norm = float(norm_depth(depth, stats))
    c_all = np.column_stack([np.full(n, d_norm, dtype=np.float32), E_norm])
    parts = []
    for i in range(0, n, chunk):
        c = torch.from_numpy(c_all[i:i + chunk]).to(dev)
        z = model.sample_fast(len(c), c, n_steps=steps)
        parts.append((z.cpu().numpy() * sd[:3] + mu[:3]).astype(np.float32))
    sp = np.concatenate(parts)
    y, dx, dy = sp[:, 0], sp[:, 1], sp[:, 2]
    r2 = dx ** 2 + dy ** 2
    bad = r2 >= 0.999999
    if bad.any():
        s = np.sqrt(0.999999 / r2[bad]); dx, dy = dx.copy(), dy.copy(); dx[bad] *= s; dy[bad] *= s
    dz = -np.sqrt(1.0 - dx ** 2 - dy ** 2)
    x = rng.uniform(-stats["x_halfwidth"], stats["x_halfwidth"], n).astype(np.float32)
    zc = np.full(n, -(stats.get("z0", Z0_CM) + depth), dtype=np.float32)
    ps7 = np.column_stack([x, y, zc, dx, dy, dz, E_phys.astype(np.float32)])
    Path(out_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as h:
        h.create_dataset("phase_space", data=ps7, compression="gzip")
        h.create_dataset("conditions", data=np.full((n, 1), depth, dtype=np.float32))
    print(f"  -> {out_h5}: {ps7.shape}")


# ------------------------------------------------------------------ main
def run_block(title, E_real, E_src, d_src, d_tgt, q_t, q_np, tau, tau_np, E_min_obs, a, max_pk_depth=3.0):
    """Fit, training multi-seed e valutazione per una coppia src->tgt. Restituisce le energie generate."""
    p, k, l = fit_pk_scipy(q_np, tau_np, max_depth=max_pk_depth)
    print(f"  [fit p,k su profondita' <= {max_pk_depth}] p={p:.4f} k={k:.2f} L1={l:.4f} MeV")
    vc = a.v_cut
    res = {}
    res["shift_k101"] = physics_transport(E_src, d_src, d_tgt, 1.77, 450.0 * 1.01, vc)
    res["fit_pk"] = physics_transport(E_src, d_src, d_tgt, p, k, vc)
    res["fit_pk (taglio E_min)"] = physics_transport(E_src, d_src, d_tgt, p, k, float(E_min_obs ** p))
    ode_runs, resq_runs = [], []
    for s in range(a.seeds):
        m = train_ode(q_t, tau, p, a.epochs, vc, 42 + s, tag=f"ODE s{s}")
        ode_runs.append(m.sample_from_source(E_src, d_src, d_tgt, tau, vc))
        m = train_res(q_t, tau, p, k, a.epochs, vc, 42 + s, tag=f"ResQ s{s}")
        resq_runs.append(m.sample_from_source(E_src, d_src, d_tgt, tau, vc))

    print(f"\n--- {title} (da {d_src} cm, Delta_d={d_tgt - d_src:.2f}) | sorgente N={len(E_src)} ---")
    for name in ("shift_k101", "fit_pk", "fit_pk (taglio E_min)"):
        eval_1d_metrics(E_real, res[name], name)
    for name, runs in (("PureNeuralODE", ode_runs), ("ResidualQ", resq_runs)):
        ws = [eval_1d_metrics(E_real, r, f"{name} s{i}") for i, r in enumerate(runs)]
        print(f"  >>> {name}: W1/std media={np.mean(ws):.4f}  min={np.min(ws):.4f}  max={np.max(ws):.4f}")
    res["ode"], res["resq"] = ode_runs[0], resq_runs[0]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/raw_phsp")
    ap.add_argument("--real_35", default="data/raw_phsp/SurfaceFile_3.5_peak_1_2_5.phsp")
    ap.add_argument("--condE_run", default="runs/cfm_condE_s42")
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--v_cut", type=float, default=0.0, help="taglio fisico in v=E^p (0 = protone fermo)")
    ap.add_argument("--only_1d", action="store_true")
    ap.add_argument("--save_dir", default="gen")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== 1. DATI (solo peak, 6 profondita' di training; 1.5 e 3.5 esclusi) ===")
    E_by = load_proton_energies(a.data_dir)
    tau = make_quantile_grid(2048, dev)
    tau_np = tau.cpu().numpy()
    q_np_all = {d: np.quantile(E, tau_np) for d, E in E_by.items()}
    q_t_all = {d: torch.from_numpy(q.astype(np.float32)).to(dev) for d, q in q_np_all.items()}
    E_min_obs = float(min(E.min() for E in E_by.values()))

    print("\n=== 2. SANITY A 2.5 cm (2.5 escluso da fit e training; NB: qui il taglio quasi non interviene) ===")
    q_np_n = {d: q for d, q in q_np_all.items() if d != 2.5}
    q_t_n = {d: q for d, q in q_t_all.items() if d != 2.5}
    run_block("hold-out 2.5 cm", E_by[2.5], E_by[2.0], 2.0, 2.5, q_t_n, q_np_n, tau, tau_np, E_min_obs, a)

    print("\n=== 3. TEST 3.5 cm (training su 0.5,1,2,2.5,3,4; fit_pk usa solo <=3.0: i modelli appresi vedono anche 4.0) ===")
    E35 = _energies_from_phsp(a.real_35)
    res = run_block("hold-out 3.5 cm", E35, E_by[3.0], 3.0, 3.5, q_t_all, q_np_all, tau, tau_np, E_min_obs, a)
    print(f"\n  Sopravvissuti reali a 3.5: {len(E35)} | frazione baseline fit_pk da 3.0: "
          f"{len(res['fit_pk']) / len(E_by[3.0]):.3f}")

    Path(a.save_dir).mkdir(exist_ok=True)
    for name, key in (("fitpk", "fit_pk"), ("shift101", "shift_k101"), ("ode", "ode"), ("resq", "resq")):
        np.save(Path(a.save_dir) / f"E_3.5_{name}.npy", res[key])

    if a.only_1d:
        print("\n[--only_1d: salto la generazione 7D]")
        return
    print("\n=== 4. PHASE SPACE 7D con cfm_condE ===")
    for name, key in (("fitpk", "fit_pk"), ("ode", "ode"), ("resq", "resq")):
        generate_full_7d_with_condE(a.condE_run, 3.5, res[key], f"{a.save_dir}/gen_3.5_neural_{name}_condE.h5")


if __name__ == "__main__":
    main()
