import os
import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pymedphys

VOXEL_MM = 4.0
FIELD_NAME = "6mv_5x5"
BASE_DIR = Path(f"outputs/outputs_condizionali_6mv_5x5")
MODELS = ["cfm", "nsf", "gan", "reference"]

print("\n=============================================================")
print(f" 🔮 INIZIO FUSIONE DELLE RUN COND: {FIELD_NAME.upper()} (100M STORIE TOTALI)")
print("=============================================================\n")

# --- 1. FUSIONE DELLE 10 RUN PARALLELE ---
for model in MODELS:
    merged_img = None
    count = 0
    print(f"📦 Elaborazione e somma 3D per modello: {model.upper()}")
    
    for run in range(1, 11):
        file_path = BASE_DIR / f"run_{run}" / f"dose_{model}_dose.mhd"
        
        if file_path.exists():
            img = sitk.ReadImage(str(file_path))
            if merged_img is None:
                merged_img = img
            else:
                merged_img = merged_img + img
            count += 1
        else:
            print(f"  ⚠️ Run {run} non trovata per file: {file_path.name}")
            
    if merged_img is not None:
        output_path = BASE_DIR / f"dose_{model}_TOTAL.mhd"
        sitk.WriteImage(merged_img, str(output_path))
        print(f"  ✓ Unito con successo ({count}/10 run) -> {output_path.name}\n")
    else:
        print(f"❌ Impossibile trovare dati per il modello: {model.upper()}\n")

# --- 2. ANALISI DOSIMETRICA DOWNSTREAM (GAMMA INDEX) ---
print("=======================================================")
print(f"    📑 CALCOLO METRICHE CLINICHE (GAMMA ANALYSIS)")
print("=======================================================")

ref_path = BASE_DIR / "dose_reference_TOTAL.mhd"
if not ref_path.exists():
    print("❌ File reference finale (_TOTAL.mhd) non trovato! Impossibile procedere.")
    sys.exit(1)

ref_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(ref_path))).astype(np.float64)
# Normalizzazione locale al valore di picco (Dmax = 100%)
ref_norm = (ref_arr / ref_arr.max()) * 100.0
axes = tuple(np.arange(s) * VOXEL_MM for s in ref_arr.shape)

for m in ["cfm", "nsf", "gan"]:
    m_path = BASE_DIR / f"dose_{m}_TOTAL.mhd"
    if not m_path.exists(): 
        print(f"⚠️ Dati globali non disponibili per il modello: {m.upper()}")
        continue
        
    m_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(m_path))).astype(np.float64)
    m_norm = (m_arr / m_arr.max()) * 100.0
    print(f"\n🧠 Analisi vs Reference per: {m.upper()}")
    
    for p in [2.0, 3.0]:
        gamma_map = pymedphys.gamma(
            axes, ref_norm, 
            axes, m_norm, 
            dose_percent_threshold=p, 
            distance_mm_threshold=p, 
            lower_percent_dose_cutoff=20.0, # Soglia di soggiacenza clinica (esclude il background a bassa dose)
            max_gamma=1.1, 
            quiet=True
        )
        valid = gamma_map[~np.isnan(gamma_map)]
        pass_rate = (valid <= 1.0).mean() * 100
        print(f"  📈 Gamma {int(p)}%/{int(p)}mm Pass Rate: {pass_rate:.2f}%")
        
print("\n=======================================================")
print(" 🎉 TUTTE LE ELABORAZIONI SONO TERMINATE CON SUCCESSO!")
print("=======================================================\n")
