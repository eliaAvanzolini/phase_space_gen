import SimpleITK as sitk
import numpy as np
from pathlib import Path

base_dir = Path("outputs/dose_validation_6mv_5x5")
ref = sitk.GetArrayFromImage(sitk.ReadImage(str(base_dir / "dose_reference_TOTAL.mhd"))).astype(np.float64)
cfm = sitk.GetArrayFromImage(sitk.ReadImage(str(base_dir / "dose_cfm_TOTAL.mhd"))).astype(np.float64)

# Normalizziamo ciascuno al rispettivo picco centrale per vedere la forma del fascio (Profile Shape)
# Troviamo le coordinate del voxel con più dose nel Reference
z_max, y_max, x_max = np.unravel_index(np.argmax(ref), ref.shape)
print(f"📍 Punto di massimo nel Reference -> Z:{z_max}, Y:{y_max}, X:{x_max}")

z_max_c, y_max_c, x_max_c = np.unravel_index(np.argmax(cfm), cfm.shape)
print(f"📍 Punto di massimo nel CFM       -> Z:{z_max_c}, Y:{y_max_c}, X:{x_max_c}")

# Estraiamo il profilo lungo l'asse Z (Profondità) passante per il centro del fascio
profilo_z_ref = ref[:, y_max, x_max]
profilo_z_cfm = cfm[:, y_max_c, x_max_c]

# Scaliamo a 100 sul rispettivo massimo del profilo per confrontare la pendenza
if profilo_z_ref.max() > 0: profilo_z_ref = (profilo_z_ref / profilo_z_ref.max()) * 100.0
if profilo_z_cfm.max() > 0: profilo_z_cfm = (profilo_z_cfm / profilo_z_cfm.max()) * 100.0

print("\n📈 CONFRONTO PROFILO IN PROFONDITÀ (PDD) - Primi 15 Voxel:")
print("------------------------------------------------------")
print("Voxel Z | REFERENCE (%) | CFM (%)")
print("------------------------------------------------------")
for i in range(min(15, len(profilo_z_ref))):
    print(f"  {i:2d}    |     {profilo_z_ref[i]:5.1f}     |   {profilo_z_cfm[i]:5.1f}")
print("------------------------------------------------------")
