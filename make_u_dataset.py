import h5py
import numpy as np
from phsp_transforms import to_features

K, P = 453.5, 1.77

with h5py.File("data/proton_depth_train.h5", "r") as h:
    ps, c = h["phase_space"][:], h["conditions"][:]

d = c[:, 0]
E_test = to_features(ps[:1000], "lin", P)[:, 3]
ecol = [j for j in range(ps.shape[1]) if np.allclose(ps[:1000, j], E_test)]
assert len(ecol) == 1, f"Colonna E non univoca o non trovata: {ecol}"
ecol = ecol[0]
print(f"Colonna E individuata all'indice: {ecol}")

ps = ps.copy()
ps[:, ecol] = ps[:, ecol].astype(np.float64) ** P + K * d

with h5py.File("data/proton_depth_train_u.h5", "w") as h:
    h.create_dataset("phase_space", data=ps.astype(np.float32))
    h.create_dataset("conditions", data=c)
    h.attrs["ecol"] = ecol

print("Dataset in u creato con successo.")
print("u min/max:", ps[:, ecol].min(), ps[:, ecol].max())
