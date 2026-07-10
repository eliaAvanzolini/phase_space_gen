import os
import SimpleITK as sitk
import numpy as np
import pymedphys

base_dir = "outputs/dose_validation"
ref_path = os.path.join(base_dir, "dose_reference_dose.mhd")
cfm_path = os.path.join(base_dir, "dose_cfm_dose.mhd")

print("\n=============================================================")
### 🚀 VERIFICA UPGRADE CFM (500 STEPS ODES) — COMPILAZIONE DATI TESI
print("=============================================================\n")

ref_img = sitk.ReadImage(ref_path)
ref_dose = sitk.GetArrayFromImage(ref_img)
cfm_img = sitk.ReadImage(cfm_path)
cfm_dose = sitk.GetArrayFromImage(cfm_img)

size = ref_img.GetSize()       
spacing = ref_img.GetSpacing() 
origin = ref_img.GetOrigin()   

coords_z = np.arange(size[2]) * spacing[2] + origin[2]
coords_y = np.arange(size[1]) * spacing[1] + origin[1]
coords_x = np.arange(size[0]) * spacing[0] + origin[0]
axes = (coords_z, coords_y, coords_x)

ref_max = np.max(ref_dose)
ref_norm = ref_dose / ref_max
mask = ref_norm > 0.10
cfm_norm = cfm_dose / np.max(cfm_dose)

# 1. TEST CRITERIO STRETTO 2% / 2mm
print("⏳ Calcolo Gamma Index [2% / 2mm] per CFM (500 steps)...")
gamma_2 = pymedphys.gamma(axes, ref_dose, axes, cfm_dose, dose_percent_threshold=2.0, distance_mm_threshold=2.0, lower_percent_dose_cutoff=10.0)
pass_2 = (np.sum(gamma_2 <= 1.0) / np.sum(~np.isnan(gamma_2))) * 100

# 2. TEST CRITERIO STANDARD 3% / 3mm
print("⏳ Calcolo Gamma Index [3% / 3mm] per CFM (500 steps)...")
gamma_3 = pymedphys.gamma(axes, ref_dose, axes, cfm_dose, dose_percent_threshold=3.0, distance_mm_threshold=3.0, lower_percent_dose_cutoff=10.0)
pass_3 = (np.sum(gamma_3 <= 1.0) / np.sum(~np.isnan(gamma_3))) * 100

diff_media = np.mean(cfm_norm[mask] - ref_norm[mask]) * 100

print("\n📊 RISULTATI FINALI CFM UPGRADE (100M Storie):")
print(f"  ➡️ Gamma Pass Rate [2% / 2mm]: {pass_2:.2f}%  (Precedente a 100 steps: 86.68%)")
print(f"  ➡️ Gamma Pass Rate [3% / 3mm]: {pass_3:.2f}%  (Precedente a 100 steps: 99.99%)")
print(f"  ➡️ Δ medio relativo (Coerente): {diff_media:+.4f}%")
print("\n=============================================================")
