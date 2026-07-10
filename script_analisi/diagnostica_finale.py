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

# Forziamo l'import delle funzioni di caricamento dal tuo modulo principale
from dose_validation_conditional import _load_cfm, _load_nsf, _load_gan_auto

PHSP2_PATH = "data/conditional_jaws_dataset.h5"
CFM_STATS_JSON = "outputs/cfm_conditional_6mv_10mv/normalization_stats.json"
GAN_STATS_JSON = "outputs/gan_conditional_6mv_10mv/normalization_stats.json"
NSF_STATS_JSON = "outputs/nsf_conditional_6mv_10mv/normalization_stats.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
COL_NAMES = ["X", "Y", "Z", "dX", "dY", "dZ", "E"]
cond_val = [6.0, 2.5, 2.5] # 6mv_5x5

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
            chunk[:, 0] = arrays["PrePosition_X"] / 10.0
            chunk[:, 1] = arrays["PrePosition_Y"] / 10.0
            chunk[:, 2] = arrays["PrePosition_Z"] / 10.0
            chunk[:, 3] = arrays["PreDirection_X"]
            chunk[:, 4] = arrays["PreDirection_Y"]
            chunk[:, 5] = arrays["PreDirection_Z"]
            chunk[:, 6] = arrays["KineticEnergy"]
            chunks.append(chunk)
    return np.concatenate(chunks, axis=0)

print("=========================================================")
print("🔬 ENGINE DI VERIFICA SPERIMENTALE: OVERFITTING & CROSS-W1")
print("=========================================================")

# 1. CARICAMENTO POOL SEEN (HDF5)
with h5py.File(PHSP2_PATH, "r") as f:
    ps_all = f["phase_space"][:]
    cond_all = f["conditions"][:]
mask_seen = np.all(np.abs(cond_all - np.array(cond_val)) < 0.1, axis=1)
real_seen = ps_all[mask_seen].astype(np.float32)
n_samples = len(real_seen)

# 2. ISOLAMENTO CHIRURGICO POOL UNSEEN (ROOT COMPLEMENT)
print("⏳ Estrazione dei fotoni ROOT mai visti (Filtro KDTree)...")
raw_pool = load_raw_gate_pool()
# Usiamo X, Y ed Energia come firma cinematica per l'esclusione geometrica
tree = cKDTree(real_seen[:, [0, 1, 6]])
distances, _ = tree.query(raw_pool[:, [0, 1, 6]], k=1)
real_unseen = raw_pool[distances > 1e-4].astype(np.float32)

print(f"📦 Campioni SEEN (HDF5 Training/Val): {n_samples}")
print(f"📦 Campioni UNSEEN (ROOT Complement isolati): {len(real_unseen)}")

from data.synthetic_linac import denormalize_phase_space

# 3. VALUTAZIONE MULTI-MODELLO SU POOL SEEN
results_w1 = {}

# --- CFM ---
with open(CFM_STATS_JSON) as f: stats_cfm = json.load(f)
with open(Path(CFM_STATS_JSON).parent / "condition_stats.json") as f: c_stats = json.load(f)
cond_norm = ((np.array(cond_val, dtype=np.float32) - np.array(c_stats["mu"])) / np.array(c_stats["sigma"])).tolist()
model_cfm = _load_cfm("outputs/cfm_conditional_6mv_10mv/best_model.pt", dim=5, device=device)
cond_tensor = torch.tensor(cond_norm, device=device).unsqueeze(0).repeat(n_samples, 1)
with torch.no_grad():
    gen_cfm = denormalize_phase_space(model_cfm.sample_fast(n_samples, cond_tensor, n_steps=30).cpu().numpy(), stats_cfm)

# --- GAN ---
with open(GAN_STATS_JSON) as f: stats_gan = json.load(f)
from dose_validation_conditional import generate_gan_sarrut
gen_gan = generate_gan_sarrut(
    {"checkpoint": "outputs/gan_conditional_6mv_10mv/best_model.pt", "stats_json": GAN_STATS_JSON, "model_type": "gan_sarrut"},
    n_samples, cond_val, device
)

# --- NSF ---
with open(NSF_STATS_JSON) as f: stats_nsf = json.load(f)
model_nsf = _load_nsf("outputs/nsf_conditional_6mv_10mv/best_model.pt", dim=5, device=device)
with torch.no_grad():
    gen_nsf = denormalize_phase_space(model_nsf.sample(n_samples, torch.tensor(cond_norm, device=device).unsqueeze(0)).cpu().numpy(), stats_nsf)

# Calcolo W1 Medie
for name, data in [("CFM (Seen)", gen_cfm), ("GAN (Seen)", gen_gan), ("NSF (Seen)", gen_nsf)]:
    w1_vals = [wasserstein_distance(real_seen[:, idx], data[:, idx]) for idx in range(7)]
    results_w1[name] = np.mean(w1_vals)

# 4. TEST DI OVERFITTING CRUCIALE PER CFM (CFM vs Unseen)
# Rigeneriamo quanti sono i campioni unseen per un confronto statistico simmetrico
n_unseen = min(len(real_unseen), 100000)
cond_tensor_unseen = torch.tensor(cond_norm, device=device).unsqueeze(0).repeat(n_unseen, 1)
with torch.no_grad():
    gen_cfm_unseen = denormalize_phase_space(model_cfm.sample_fast(n_unseen, cond_tensor_unseen, n_steps=30).cpu().numpy(), stats_cfm)

w1_unseen_vals = [wasserstein_distance(real_unseen[:n_unseen, idx], gen_cfm_unseen[:, idx]) for idx in range(7)]
results_w1["CFM (Unseen Data)"] = np.mean(w1_unseen_vals)

print("\n📊 =========================================================")
print("📊 VERDETTO COMPARATIVO FINALE DELLA DISTANZA W1 (6mv_5x5):")
print("=========================================================")
for label, score in results_w1.items():
    print(f" 🟢 {label:<22} -> W1 Media: {score:.4f}")
print("=========================================================\n")
