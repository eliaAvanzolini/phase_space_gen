import h5py
import numpy as np
from scipy.stats import spearmanr
from phsp_transforms import to_features

rng = np.random.default_rng(0)
Q = [1, 5, 25, 50, 75, 95]

def feats(ps):
    return to_features(ps, "lin", 1.77)  # y, dx, dy, E

def read_phsp(h):
    key = "phase_space" if "phase_space" in h else list(h.keys())[0]
    return h[key][:]

with h5py.File("data/proton_depth_train.h5", "r") as h:
    ps, d = h["phase_space"][:], h["conditions"][:, 0]

f = feats(ps[:1000])
print("E media primi 1000 (deve essere in MeV, 0-70):", f[:, 3].mean())

print("\n== REALI: Spearman(theta, E) e quantili di E ==")
real_q = {}
for dv in np.unique(d):
    mask = (d == dv)
    x = feats(ps[mask])
    x = x[rng.choice(len(x), min(200_000, len(x)), replace=False)]
    th = np.hypot(x[:, 1], x[:, 2])
    rho = spearmanr(th, x[:, 3])[0]
    real_q[dv] = np.percentile(x[:, 3], Q)
    print(f"d={dv:4.2f}  rho={rho:+.3f}  q={np.round(real_q[dv], 2)}")

print("\n== SINTETICI ==")
for p, dv in [("data/synth_3.25.h5", 3.25), ("data/synth_3.75.h5", 3.75)]:
    with h5py.File(p, "r") as h:
        x = feats(read_phsp(h))
    x = x[rng.choice(len(x), min(200_000, len(x)), replace=False)]
    th = np.hypot(x[:, 1], x[:, 2])
    print(f"d={dv:4.2f}  rho={spearmanr(th, x[:, 3])[0]:+.3f}  q={np.round(np.percentile(x[:, 3], Q), 2)}")
    w = (dv - 3.0) / 1.0
    print(f"         interp 3.0-4.0: q={np.round((1 - w) * real_q[3.0] + w * real_q[4.0], 2)}")
