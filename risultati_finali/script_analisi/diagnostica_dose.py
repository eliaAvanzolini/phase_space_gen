import SimpleITK as sitk
import numpy as np
from pathlib import Path

BASE_DIR = Path("outputs/outputs_condizionali_6mv_5x5")
MODELS = ["reference", "cfm", "nsf", "gan"]

print("\n=======================================================")
# PROFILO GEOMETRICO DELLE MATRICI DI DOSE 3D
print(" 🔬 AUTOPSIA GEOMETRICA DEI VOLUMI DI DOSE")
print("=======================================================\n")

for m in MODELS:
    p = BASE_DIR / f"dose_{m}_TOTAL.mhd"
    if not p.exists():
        print(f"❌ File non trovato per {m.upper()}")
        continue
        
    img = sitk.ReadImage(str(p))
    arr = sitk.GetArrayFromImage(img).astype(np.float64)
    
    max_val = arr.max()
    max_idx = np.unravel_index(arr.argmax(), arr.shape)
    
    # Calcoliamo la dose integrata totale per vedere se l'energia prodotta è simile
    total_energy = arr.sum()
    
    # Estraiamo il profilo centrale lungo l'asse Z (profondità) alla coordinata del picco del reference
    # Nota: assumiamo che il centro del phantom sia stabile
    print(f"🧠 Modello: {m.upper()}")
    print(f"  · Dose Massima Assoluta: {max_val:.2e}")
    print(f"  · Posizione del Voxel di Picco (Z, Y, X): {max_idx}")
    print(f"  · Energia Totale Deposta (Somma voxel): {total_energy:.2e}")
    
    # Stampiamo un mini-profilo della dose lungo l'asse centrale per capire la forma
    z_center, y_center, x_center = max_idx
    profilo_profondita = arr[:, y_center, x_center]
    print(f"  · Primi 5 voxel in profondità Z: {profilo_profondita[:5]}")
    print("-" * 50)
