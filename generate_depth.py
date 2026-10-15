"""
generate_depth.py
=================
Genera un phase space a una profondità qualsiasi (anche non vista in training) con un modello addestrato.

    python generate_depth.py --run runs/cfm_depth_v1 --depth 3.25 --n 1000000 --out gen_3.25.h5
    python generate_depth.py --run runs/cfm_depth_v1 --depth 3.25 --n 1000000 --out gen_3.25.phsp   # ASCII TOPAS

Fuori dall'intervallo [d_min, d_max] di training è estrapolazione: viene stampato un avviso.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from phsp_transforms import destandardize_at, features_to_7d, load_stats, norm_depth
from read_topas_phsp import write_topas_ascii


def load_model(run_dir, ckpt=None, device=None):
    import json
    from depth_models import build, load_weights
    run = Path(run_dir); stats = load_stats(run / "stats.json"); cfg = json.load(open(run / "run.json"))
    kind = cfg.get("model", "cfm")                         # i vecchi run (solo CFM) non hanno la chiave "model"
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = load_weights(build(kind, cfg, len(stats["channels"]), dev), kind, run / (ckpt or ("last.pt" if kind == "gan" else "best.pt")), dev)
    return m, kind, stats, dev


@torch.no_grad()
def generate(run_dir, depth, n, steps=200, chunk=None, ckpt=None, seed=0, dopri5=False):
    from depth_models import sample
    model, kind, stats, dev = load_model(run_dir, ckpt)
    chunk = chunk or (200_000 if kind == "nsf" else 500_000)
    if not (stats["d_min"] - 1e-6 <= depth <= stats["d_max"] + 1e-6):
        print(f"[ATTENZIONE] depth={depth} fuori da [{stats['d_min']}, {stats['d_max']}]: estrapolazione")
    print(f"modello: {kind}")
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    c_val = float(norm_depth(depth, stats)); out = []
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        c = torch.full((m, 1), c_val, device=dev)
        z = sample(model, kind, c, steps=steps, dopri5=dopri5)
        out.append(features_to_7d(destandardize_at(z.cpu().numpy(), depth, stats), depth, stats, rng))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--depth", type=float, required=True)
    ap.add_argument("--n", type=int, default=1_000_000); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200); ap.add_argument("--dopri5", action="store_true")
    ap.add_argument("--ckpt", default=None, help="default: best.pt (cfm, nsf), last.pt (gan)"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ps = generate(a.run, a.depth, a.n, a.steps, ckpt=a.ckpt, seed=a.seed, dopri5=a.dopri5)
    if a.out.endswith(".phsp"):
        write_topas_ascii(ps, a.out)
    else:
        import h5py
        with h5py.File(a.out, "w") as h:
            h.create_dataset("phase_space", data=ps, compression="gzip")
            h.create_dataset("conditions", data=np.full((len(ps), 1), a.depth, np.float32))
    print(f"scritto {a.out}: {ps.shape}")


if __name__ == "__main__":
    main()
