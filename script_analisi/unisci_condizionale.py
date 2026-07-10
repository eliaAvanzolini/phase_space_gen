import os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pymedphys

VOXEL_MM = 4.0

def merge_and_analyse(field_name="6mv_5x5"):
    base_dir = Path(f"outputs/dose_validation_{field_name}")
    
    # Mappiamo i nomi reali sputati da GATE (nome_subtask_dose.mhd)
    models = ["reference", "cfm", "nsf", "gan"]
    for m in models:
        f_path = base_dir / f"dose_{m}_dose.mhd"
        if f_path.exists():
            # Creiamo il file _TOTAL leggendo direttamente quello buono di GATE
            sitk.WriteImage(sitk.ReadImage(str(f_path)), str(base_dir / f"dose_{m}_TOTAL.mhd"))
            
    ref_path = base_dir / "dose_reference_TOTAL.mhd"
    if not ref_path.exists():
        print("❌ File reference finale (_TOTAL.mhd) non trovato!")
        return

    ref_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(ref_path))).astype(np.float64)
    ref_norm = (ref_arr / ref_arr.max()) * 100.0
    axes = tuple(np.arange(s) * VOXEL_MM for s in ref_arr.shape)

    print("\n=======================================================")
    print(f"    📑 RISULTATI DOSIMETRICI FINALI COND: {field_name.upper()}")
    print("=======================================================")
    
    for m in ["cfm", "nsf", "gan"]:
        m_path = base_dir / f"dose_{m}_TOTAL.mhd"
        if not m_path.exists(): 
            print(f"⚠️ Dati non ancora disponibili per il modello: {m.upper()}")
            continue
        m_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(m_path))).astype(np.float64)
        m_norm = (m_arr / m_arr.max()) * 100.0
        print(f"\n🧠 Modello: {m.upper()}")
        for p in [2.0, 3.0]:
            gamma_map = pymedphys.gamma(axes, ref_norm, axes, m_norm, dose_percent_threshold=p, distance_mm_threshold=p, lower_percent_dose_cutoff=20.0, max_gamma=1.1, quiet=True)
            valid = gamma_map[~np.isnan(gamma_map)]
            print(f"  📈 Gamma {int(p)}%/{int(p)}mm Pass Rate: {(valid <= 1.0).mean() * 100:.2f}%")
    print("=======================================================\n")

if __name__ == "__main__":
    merge_and_analyse()
