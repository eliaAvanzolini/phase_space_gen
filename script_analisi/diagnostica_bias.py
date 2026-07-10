import h5py
import numpy as np
import torch
import json
from pathlib import Path
from scipy.stats import wasserstein_distance

# Importiamo i metodi di caricamento dal tuo script di validazione esistente
from dose_validation_conditional import _load_cfm

PHSP2_PATH = "data/conditional_jaws_dataset.h5"
CFM_CKPT = "outputs/cfm_conditional_6mv_10mv/best_model.pt"
STATS_JSON = "outputs/cfm_conditional_6mv_10mv/normalization_stats.json"

with open(STATS_JSON) as f: stats = json.load(f)
with open(Path(STATS_JSON).parent / "condition_stats.json") as f: cond_stats = json.load(f)

mu_c = np.array(cond_stats["mu"], dtype=np.float32)
sig_c = np.array(cond_stats["sigma"], dtype=np.float32)

# FIX: Leggiamo la dimensione reale del modello dal file di normalizzazione (sarà 5D)
dim = len(stats.get("col_names", ["x", "y", "theta", "phi", "E"]))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = _load_cfm(CFM_CKPT, dim=dim, device=device)

print("=========================================================")
print("🔬 DIAGNOSTICA SUL BIAS CONDIZIONALE DI BORDO (W1 DISTANCE)")
print("=========================================================")

# Caricamento del dataset reale
with h5py.File(PHSP2_PATH, "r") as f:
    ps_all = f["phase_space"][:]
    cond_all = f["conditions"][:]

COL_NAMES = ["X", "Y", "Z", "dX", "dY", "dZ", "E"]

for label, cond_val in [("⚠️ CAMPO ESTREMO (6mv_5x5)", [6.0, 2.5, 2.5]), 
                        ("🍏 CAMPO CENTRALE (6mv_10x10)", [6.0, 5.0, 5.0])]:
    
    # 1. Filtro dati reali
    mask = np.all(np.abs(cond_all - np.array(cond_val)) < 0.1, axis=1)
    real_samples = ps_all[mask]
    n_samples = len(real_samples)
    
    if n_samples == 0:
        print(f"❌ Nessun dato reale trovato per {label}")
        continue
        
    # 2. Generazione CFM corrispondente
    cond_norm = ((np.array(cond_val, dtype=np.float32) - mu_c) / sig_c).tolist()
    cond_tensor = torch.tensor(cond_norm, device=device).unsqueeze(0).repeat(n_samples, 1)
    
    with torch.no_grad():
        # Usiamo 30 step (visto che abbiamo dimostrato che bastano!)
        gen_norm = model.sample_fast(n_samples, cond_tensor, n_steps=30).cpu().numpy()
    
    # Denormalizzazione speculare per avere unità fisiche reali (restituisce 7D)
    from data.synthetic_linac import denormalize_phase_space
    gen_samples = denormalize_phase_space(gen_norm, stats)
    
    print(f"\n🔹 {label} | Campioni analizzati: {n_samples}")
    print(f"{'Variabile':<12} | {'Real Mean':<10} | {'Gen Mean':<10} | {'W1 Distance':<12}")
    print("-" * 55)
    
    w1_total = 0.0
    for idx, col in enumerate(COL_NAMES):
        w1 = wasserstein_distance(real_samples[:, idx], gen_samples[:, idx])
        w1_total += w1
        print(f"{col:<12} | {real_samples[:, idx].mean():10.3f} | {gen_samples[:, idx].mean():10.3f} | {w1:10.4f}")
    
    print("-" * 55)
    print(f"🏆 W1 MEDIA CONFIGURAZIONE: {w1_total / len(COL_NAMES):.4f}\n")

print("=========================================================")
