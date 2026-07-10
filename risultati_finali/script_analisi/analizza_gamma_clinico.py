import json
import pathlib
import numpy as np
import SimpleITK as sitk
import pymedphys

out = pathlib.Path("outputs/dose_validation")
models = ["cfm", "nsf", "gan"]
VOXEL_MM = 4.0  # Spaziatura uniforme dei voxel in mm

print("============================================================")
print(" 🩺 ANALISI GAMMA INDEX 3D CLINICA (Con Ricerca Spaziale DTA)")
print("============================================================")

# 1. Carica il volume di riferimento (MC Reference)
ref_path = out / "dose_reference_dose.mhd"
if not ref_path.exists():
    ref_path = out / "dose_reference-dose.mhd"

print(f"Caricamento Reference: {ref_path.name}")
ref_img = sitk.ReadImage(str(ref_path))
ref = sitk.GetArrayFromImage(ref_img).astype(np.float64)

# Normalizzazione al proprio massimo (Dmax) come da protocollo
D_max_ref = ref.max()
ref_norm = (ref / D_max_ref) * 100.0

# Definizione degli assi spaziali 3D richiesti da pymedphys (Z, Y, X)
# Nota: sitk legge in formato [Z, Y, X]
axes = tuple(np.arange(s) * VOXEL_MM for s in ref.shape)

for m in models:
    model_path = out / f"dose_{m}_dose.mhd"
    if not model_path.exists():
        model_path = out / f"dose_{m}-dose.mhd"
        
    if not model_path.exists():
        print(f"❌ File dose per {m.upper()} non trovato, salto.")
        continue
        
    print(f"\nCalcolo Gamma Index 3D per {m.upper()}...")
    model_img = sitk.ReadImage(str(model_path))
    model = sitk.GetArrayFromImage(model_img).astype(np.float64)
    
    # Normalizzazione al proprio massimo
    model_norm = (model / model.max()) * 100.0
    
    # Lancio dell'algoritmo Gamma 3D di pymedphys
    # Criteri rigidi: 2% Dose Difference, 2mm Distance-to-Agreement, 10% Threshold
    gamma_map = pymedphys.gamma(
        axes, ref_norm,
        axes, model_norm,
        dose_percent_threshold=2.0,
        distance_mm_threshold=2.0,
        lower_percent_dose_cutoff=10.0,
        quiet=True
    )
    
    # Estrazione dei voxel validi (escludendo i NaN fuori dalla soglia del 10%)
    valid_gamma = gamma_map[~np.isnan(gamma_map)]
    pass_rate = float((valid_gamma <= 1.0).mean() * 100)
    
    print(f"  ✅ Risultato {m.upper()}:")
    print(f"     Voxel valutati:  {len(valid_gamma):,}")
    # Il valore di gamma medio esprime la bontà complessiva dell'accordo
    print(f"     Gamma Medio:     {np.mean(valid_gamma):.4f}") 
    print(f"     💥 GAMMA PASS RATE: {pass_rate:.2f}%")
    
    # Salva il report aggiornato nel JSON esistente per non perdere i vettori dei grafici
    json_p = out / f"dose_comparison_{m}.json"
    if json_p.exists():
        with open(json_p, "r") as f:
            data = json.load(f)
        data["gamma_pass_rate_pct"] = pass_rate
        data["gamma_dta_mm"] = 2.0
        with open(json_p, "w") as f:
            json.dump(data, f, indent=2)

print("\n📊 Elaborazione completata!")
