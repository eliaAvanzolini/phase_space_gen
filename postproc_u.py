import h5py
import numpy as np
import sys

K, P = 453.5, 1.77
src = sys.argv[1]
dst = sys.argv[2]
D = float(sys.argv[3])

with h5py.File(src, "r") as h:
    key = "phase_space" if "phase_space" in h else list(h.keys())[0]
    x = h[key][:]

with h5py.File("data/proton_depth_train_u.h5", "r") as h:
    ecol = int(h.attrs["ecol"])

v = x[:, ecol].astype(np.float64) - K * D
keep = v > 0
frazione = keep.mean()
print(f"Sopravvissuti al taglio (v > 0): {frazione:.3f} ({keep.sum()} su {len(keep)})")

x_surv = x[keep].copy()
x_surv[:, ecol] = (v[keep] ** (1.0 / P)).astype(x.dtype)

with h5py.File(dst, "w") as h:
    h.create_dataset(key, data=x_surv)

print(f"File processato salvato in: {dst} con shape {x_surv.shape}")
