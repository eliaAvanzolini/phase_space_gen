"""
Valutazione oracle a 3.5 cm di piu' run 3D, con medie per gruppo.
Uso: python eval_3d_runs.py --group base runs/cfm_condE_s42 runs/cfm_condE_s43 runs/cfm_condE_s44 \
                            --group flip runs/cfm_condE_flip_s43 runs/cfm_condE_flip_s44 \
                            --group scale runs/cfm_condE_scale_flip_s43 runs/cfm_condE_scale_flip_s44
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance

import read_topas_phsp as rtp
from phsp_transforms import stats_at, e_fwd, norm_depth
from generate_depth_condE import load_condE_model
from scale_net import ScaleNet, sigma

REAL = "data/raw_phsp/SurfaceFile_3.5_peak_1_2_5.phsp"
BANDS = [(0, 10, "lenti<10"), (10, 18, "10-18"), (18, 80, "veloci>=18")]


def load_real(path):
    df = rtp.read_phsp(str(path))
    if "pdg" in df.columns:
        df = df[df["pdg"] == 2212]
    ps = rtp.to_7d(df).astype(np.float64)
    return ps[ps[:, 6] > 0]


@torch.no_grad()
def gen(run, E, steps=200, chunk=500_000):
    model, stats, dev = load_condE_model(run)
    torch.manual_seed(0)
    mu, sd = stats_at(stats, 3.5)
    fE = e_fwd(E, stats.get("etrans", "lin"), stats.get("p", 1.77))
    En = ((fE - mu[3]) / sd[3]).astype(np.float32)
    dn = float(norm_depth(3.5, stats))
    cond = np.column_stack([np.full(len(E), dn, np.float32), En])
    net = None
    sp = Path(run) / "scale_net.pt"
    if sp.exists():
        net = ScaleNet(); net.load_state_dict(torch.load(sp, map_location="cpu")); net.eval()
    out = []
    for i in range(0, len(E), chunk):
        c = torch.from_numpy(cond[i:i + chunk]).to(dev)
        z = model.sample_fast(len(c), c, n_steps=steps).cpu().numpy()
        y = z[:, 0] * sd[0] + mu[0]
        if net is not None:
            ang = z[:, 1:3] * sigma(net, E[i:i + chunk], 3.5, device=dev)
        else:
            ang = z[:, 1:3] * sd[1:3] + mu[1:3]
        out.append(np.column_stack([y, ang]))
    return np.concatenate(out)


def metrics(real, g, E):
    r = real[:, [1, 3, 4]]
    m = {}
    for i, n in enumerate(["y", "dx", "dy"]):
        m["w1_" + n] = wasserstein_distance(r[:, i], g[:, i]) / r[:, i].std()
        m["dm_" + n] = (g[:, i].mean() - r[:, i].mean()) / r[:, i].std()
    cr, cg = np.corrcoef(r.T), np.corrcoef(g.T)
    m["c_ydy"] = cg[0, 2]; m["c_dxdy"] = cg[1, 2]; m["maxdc"] = np.abs(cg - cr).max()
    for lo, hi, nm in BANDS:
        s = (E >= lo) & (E < hi)
        for i, n in ((1, "dx"), (2, "dy")):
            m[f"{nm}_{n}"] = (g[s, i].std() - r[s, i].std()) / r[s, i].std()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="+", action="append", required=True)
    a = ap.parse_args()
    real = load_real(REAL)
    E = real[:, 6]
    res = {}
    print(f"{'run':38s} {'W1 y':>7s} {'W1 dx':>7s} {'W1 dy':>7s} {'dm dx':>7s} {'dm dy':>7s} {'c_ydy':>6s} {'maxdc':>6s} | dstd dx/dy per banda (lenti, 10-18, veloci)")
    for grp in a.group:
        name, runs = grp[0], grp[1:]
        res[name] = []
        for r in runs:
            m = metrics(real, gen(r, E), E)
            res[name].append(m)
            bands = "  ".join(f"{100*m[f'{nm}_dx']:+5.1f}/{100*m[f'{nm}_dy']:+5.1f}%" for _, _, nm in BANDS)
            print(f"{r:38s} {m['w1_y']:7.4f} {m['w1_dx']:7.4f} {m['w1_dy']:7.4f} {m['dm_dx']:+7.4f} {m['dm_dy']:+7.4f} {m['c_ydy']:6.3f} {m['maxdc']:6.4f} | {bands}", flush=True)
    print("\n=== MEDIA [min-max] PER GRUPPO ===")
    for name, ms in res.items():
        def f(k):
            v = [m[k] for m in ms]
            return f"{np.mean(v):.4f} [{min(v):.4f}-{max(v):.4f}]"
        print(f"{name:8s} (n={len(ms)}) W1 dx {f('w1_dx')} | W1 dy {f('w1_dy')} | W1 y {f('w1_y')} | corr(y,dy) {np.mean([m['c_ydy'] for m in ms]):.3f}")


if __name__ == "__main__":
    main()
