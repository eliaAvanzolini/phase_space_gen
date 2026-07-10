import os
import sys
import glob
import h5py
import json
import torch
import numpy as np
from pathlib import Path
from scipy.stats import wasserstein_distance
from scipy.spatial import cKDTree

# Import dei metodi dal modulo principale
from dose_validation_conditional import _load_cfm

PHSP2_PATH = "data/conditional_jaws_dataset.h5"
CFM_CKPT = "outputs/cfm_conditional_6mv_10mv/best_model.pt"
CFM_STATS_JSON = "outputs/cfm_conditional_6mv_10mv/normalization_stats.json"

device = "cuda" if torch.torch.cuda.is_available() else "cpu"
COL_NAMES = ["X", "Y", "Z", "dX", "dY", "dZ", "E"]
cond_val = [6.0, 2.5, 2.5] 

def load_raw_gate_pool():
    files = sorted(glob.glob("outputs/gate_jaw/6mv_5x5/6mv_5x5_phsp_part*.root"))
    if not files:
        print("❌ Errore: File ROOT primari non trovati!")
        sys.exit(1)
    import uproot
    gate_branches = ["PrePosition_X", "PrePosition_Y", "PrePosition_Z", "PreDirection_X", "PreDirection_Y", "PreDirection_Z", "KineticEnergy"]
    chunks = []
    for fpath in files:
        with uproot.open(fpath) as f:
            if not f.keys(): continue
            tree = f[f.keys()[0]]
            arrays = tree.arrays(gate_branches, library="np")
            chunk = np.zeros((len(arrays["KineticEnergy"]), 7), dtype=np.float32)
            chunk[:, 0] = arrays["PrePosition_X"] / 10.0  # mm -> cm
            chunk[:, 1] = arrays["PrePosition_Y"] / 10.0
            chunk[:, 2] = arrays["PrePosition_Z"] / 10.0
            chunk[:, 3] = arrays["PreDirection_X"]
            chunk[:, 4] = arrays["PreDirection_Y"]
            chunk[:, 5] = arrays["PreDirection_Z"]
            chunk[:, 6] = arrays["KineticEnergy"]
            chunks.append(chunk)
    return np.concatenate(chunks, axis=0)

print("=========================================================")
print("🔬 CONTROLLO BIAS DI CAMPIONAMENTO ED ESTRAZIONE UNSEEN 7D")
print("=========================================================")

# 1. Caricamento dati visti (HDF5)
with h5py.File(PHSP2_PATH, "r") as f:
    ps_all = f["phase_space"][:]
    cond_all = f["conditions"][:]
mask_seen = np.all(np.abs(cond_all - np.array(cond_val)) < 0.1, axis=1)
real_seen = ps_all[mask_seen].astype(np.float32)

# 2. Caricamento Pool RAW completo
raw_pool = load_raw_gate_pool()

print(f"📦 Pool RAW Totale da file ROOT: {len(raw_pool)}")
print(f"📦 Sottoinsieme SEEN da HDF5:      {len(real_seen)}")
print(f"📉 Complemento teorico atteso:     {len(raw_pool) - len(real_seen)}")

# 3. Estrazione con KDTree a 7 Dimensioni Complete
print("\n⏳ Costruzione albero spaziale 7D per isolamento perfetto...")
# Usiamo tutte le colonne (X, Y, Z, dX, dY, dZ, E) per azzerare i falsi positivi
tree_7d = cKDTree(real_seen)
distances, _ = tree_7d.query(raw_pool, k=1)

# Applichiamo una tolleranza strettissima per l'identità numerica float32
real_unseen_7d = raw_pool[distances > 1e-5].astype(np.float32)
print(f"🎯 Campioni UNSEEN estratti in 7D: {len(real_unseen_7d)}")

# Ricreiamo anche il vecchio pool distorto 3D per stampare la verifica intermedia
tree_3d = cKDTree(real_seen[:, [0, 1, 6]])
dist_3d, _ = tree_3d.query(raw_pool[:, [0, 1, 6]], k=1)
real_unseen_3d = raw_pool[dist_3d > 1e-4].astype(np.float32)

print("\n📊 -----------------------------------------------------")
print("📊 VERIFICA INTERMEDIA RAPIDA DELLA DISTORSIONE STATISTICA:")
print("-----------------------------------------------------")
print(f"Pool RAW Completo  - X: mean={raw_pool[:,0].mean():.4f} std={raw_pool[:,0].std():.4f}")
print(f"Unseen (Vecchio 3D)- X: mean={real_unseen_3d[:,0].mean():.4f} std={real_unseen_3d[:,0].std():.4f}")
print(f"Unseen (Nuovo 7D)  - X: mean={real_unseen_7d[:,0].mean():.4f} std={real_unseen_7d[:,0].std():.4f}")
print("-" * 53)
print(f"Pool RAW Completo  - E: mean={raw_pool[:,6].mean():.4f} std={raw_pool[:,6].std():.4f}")
print(f"Unseen (Vecchio 3D)- E: mean={real_unseen_3d[:,6].mean():.4f} std={real_unseen_3d[:,6].std():.4f}")
print(f"Unseen (Nuovo 7D)  - E: mean={real_unseen_7d[:,6].mean():.4f} std={real_unseen_7d[:,6].std():.4f}")
print("-----------------------------------------------------")

# 4. Calcolo della vera W1 del CFM contro il vero set Unseen
with open(CFM_STATS_JSON) as f: stats_cfm = json.load(f)
with open(Path(CFM_STATS_JSON).parent / "condition_stats.json") as f: c_stats = json.load(f)
cond_norm = ((np.array(cond_val, dtype=np.float32) - np.array(c_stats["mu"])) / np.array(c_stats["sigma"])).tolist()

model_cfm = _load_cfm(CFM_CKPT, dim=5, device=device)
n_eval = min(len(real_unseen_7d), 100000)
cond_tensor = torch.tensor(cond_norm, device=device).unsqueeze(0).repeat(n_eval, 1)

with torch.no_grad():
    gen_norm = model_cfm.sample_fast(n_eval, cond_tensor, n_steps=30).cpu().numpy()

from data.synthetic_linac import denormalize_phase_space
gen_cfm = denormalize_phase_space(gen_norm, stats_cfm)

w1_vals = [wasserstein_distance(real_unseen_7d[:n_eval, idx], gen_cfm[:, idx]) for idx in range(7)]

print("\n🔥 =========================================================")
print(f"🔥 VERA W1 MEDIA CFM (Contro dati Unseen 7D): {np.mean(w1_vals):.4f}")
print("=========================================================\n")
