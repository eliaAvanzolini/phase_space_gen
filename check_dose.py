import SimpleITK as sitk
import numpy as np

# Carica il file MHD (legge automaticamente il file .raw associato)
mhd_path = "outputs/dose_nsf_TEST/dose_linac_6MV_nsf_TEST_dose.mhd"
image = sitk.ReadImage(mhd_path)

# Converte in un array NumPy
dose_matrix = sitk.GetArrayFromImage(image)

print("=== VERIFICA GEOMETRIA DOSE ===")
print(f"Dimensioni della matrice (Z, Y, X): {dose_matrix.shape}")
print(f"Dose Massima depositata: {np.max(dose_matrix)}")
print(f"Dose Media depositata: {np.mean(dose_matrix)}")
print(f"Numero di voxel colpiti (Dose > 0): {np.sum(dose_matrix > 0)} su {dose_matrix.size}")

if np.max(dose_matrix) > 0:
    print("\n✅ PIPELINE CORRETTA! Il file contiene dati reali di dose.")
else:
    print("\n❌ ERRORE: La matrice è vuota (tutti zeri).")