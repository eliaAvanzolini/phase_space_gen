"""depth_data.py (numpy) - preparazione dati comune a train_depth_{cfm,nsf,gan}: stessa standardizzazione per profondità."""
import h5py
import numpy as np
from phsp_transforms import fit_depth_stats, norm_depth, standardize_for_training, to_features


def add_data_args(ap):
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--max_samples", type=int, default=None, help="per smoke test")
    ap.add_argument("--etrans", choices=["log", "lin", "sqrt", "Ep"], default="lin", help="trasformazione di E (default lin = run R2)")
    ap.add_argument("--stat", choices=["moments", "robust"], default="moments")
    ap.add_argument("--interp", choices=["hybrid", "pchip_Ep", "pchip", "linear"], default="hybrid")
    ap.add_argument("--norm", choices=["depth", "global"], default="depth")
    ap.add_argument("--p", type=float, default=1.77)


def load_standardized(a, rng):
    """-> X (N,4) standardizzato, C (N,1) profondità in [0,1], stats (dict, da salvare in stats.json)."""
    with h5py.File(a.data, "r") as h:
        ps7 = h["phase_space"][:]; depth = h["conditions"][:, 0]
    if a.max_samples and len(ps7) > a.max_samples:
        idx = rng.choice(len(ps7), a.max_samples, replace=False); ps7, depth = ps7[idx], depth[idx]
    print(f"dati: {ps7.shape}, profondità: {np.unique(depth)}")
    feat = to_features(ps7, a.etrans, a.p)
    stats = fit_depth_stats(feat, depth, mode=a.norm, interp=a.interp, stat=a.stat, etrans=a.etrans, p=a.p)
    X = standardize_for_training(feat, depth, stats)
    C = norm_depth(depth, stats)[:, None]
    print(f"standardizzazione: {a.norm}, stat: {a.stat}, etrans: {a.etrans}, interp: {a.interp}")
    print(f"canali standardizzati: min {X.min(0).round(1).tolist()} max {X.max(0).round(1).tolist()}")
    return X, C, stats
