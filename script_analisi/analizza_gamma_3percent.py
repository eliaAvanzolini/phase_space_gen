import os
import SimpleITK as sitk
import numpy as np
import pymedphys

base_dir = "outputs/dose_validation"
models = ["cfm", "nsf", "gan"]

print("\n=============================================================")
print(" 📊 RICALCOLO GAMMA INDEX CLINICO (3.0% / 3.0mm — Normalizzazione Coerente)")
print(" 🧮 STATISTICA: 100 MILIONI DI STORIE (FILE FUSI)")
print("=============================================================\n")

ref_path = os.path.join(base_dir, "dose_reference_dose.mhd")
if not os.path.exists(ref_path):
    print(f"❌ Errore: File di riferimento non trovato in {ref_path}")
    exit(1)

ref_img = sitk.ReadImage(ref_path)
ref_dose = sitk.GetArrayFromImage(ref_img)

size = ref_img.GetSize()       
spacing = ref_img.GetSpacing() 
origin = ref_img.GetOrigin()   

coords_z = np.arange(size[2]) * spacing[2] + origin[2]
coords_y = np.arange(size[1]) * spacing[1] + origin[1]
coords_x = np.arange(size[0]) * spacing[0] + origin[0]
axes = (coords_z, coords_y, coords_x)

# Prepariamo il reference normalizzato al suo max per la maschera e il delta coerente
ref_max = np.max(ref_dose)
ref_norm = ref_dose / ref_max
mask = ref_norm > 0.10

for model in models:
    model_path = os.path.join(base_dir, f"dose_{model}_dose.mhd")
    if not os.path.exists(model_path):
        print(f"⚠️ Warning: Mappa di dose per {model.upper()} non trouvata.")
        continue
        
    img = sitk.ReadImage(model_path)
    dose = sitk.GetArrayFromImage(img)
    
    print(f"⏳ Calcolo Gamma Index 3D per il modello: {model.upper()}...")
    
    gamma = pymedphys.gamma(
        axes, ref_dose,
        axes, dose,
        dose_percent_threshold=3.0,
        distance_mm_threshold=3.0,
        lower_percent_dose_cutoff=10.0
    )
    
    valid_voxels = np.sum(~np.isnan(gamma))
    passed_voxels = np.sum(gamma <= 1.0)
    pass_rate = (passed_voxels / valid_voxels) * 100
    
    # FORMULA CORRETTA: Normalizzazione separata identica al codice a 2%
    model_norm = dose / np.max(dose)
    diff_media = np.mean(model_norm[mask] - ref_norm[mask]) * 100
    
    print(f"  👉 Voxel valutati (>10% Dmax): {valid_voxels:,}")
    print(f"  👉 Δ medio relativo (Coerente): {diff_media:+.4f}%")
    print(f"  💥 GAMMA PASS RATE CLINICO: {pass_rate:.2f}%  [3.0% / 3.0mm]\n")

print("=============================================================")
print(" ✅ Analisi coerente al 3% completata!")
print("=============================================================")
