import os
import SimpleITK as sitk
import numpy as np

BASE_DIR = "outputs/dose_validation"
FIELDS = ["6mv_5x5", "6mv_10x10", "6mv_20x20", "10mv_5x5", "10mv_10x10", "10mv_20x20"]

def main():
    print("=" * 90)
    print("ANALISI PROFILI 1D SULL'ASSE CENTRALE (CAX)")
    print("=" * 90)

    for field in FIELDS:
        ref_path = os.path.join(BASE_DIR, field, "dose_reference_dose.mhd")
        cfm_path = os.path.join(BASE_DIR, field, "dose_cfm_dose.mhd")
        nsf_path = os.path.join(BASE_DIR, field, "dose_nsf_dose.mhd")

        if not os.path.exists(ref_path):
            continue

        # SimpleITK legge in formato (Z, Y, X)
        ref = sitk.GetArrayFromImage(sitk.ReadImage(ref_path)).astype(np.float64)
        cfm = sitk.GetArrayFromImage(sitk.ReadImage(cfm_path)).astype(np.float64)
        nsf = sitk.GetArrayFromImage(sitk.ReadImage(nsf_path)).astype(np.float64)

        nz, ny, nx = ref.shape
        cy, cx = ny // 2, nx // 2  # Centro del campo in X e Y

        # Profilo di dose in profondità lungo Z (Asse Centrale)
        pdd_ref = ref[:, cy, cx]
        pdd_cfm = cfm[:, cy, cx]
        pdd_nsf = nsf[:, cy, cx]

        # Troviamo la Dmax sull'asse centrale per la Reference
        z_dmax_idx = np.argmax(pdd_ref)
        cax_dmax_ref = pdd_ref[z_dmax_idx]

        # Normalizziamo tutti e 3 i volumi sulla Dmax dell'ASSE CENTRALE della Reference
        # (Questo elimina ogni problema di outlier e uniforma la scala fisica!)
        ref_cax_norm = (ref / cax_dmax_ref) * 100.0
        cfm_cax_norm = (cfm / cax_dmax_ref) * 100.0
        nsf_cax_norm = (nsf / cax_dmax_ref) * 100.0

        # Calcoliamo il Delta medio sull'asse centrale a profondità > dmax (zona di decadimento)
        pdd_ref_n = ref_cax_norm[:, cy, cx]
        pdd_cfm_n = cfm_cax_norm[:, cy, cx]
        pdd_nsf_n = nsf_cax_norm[:, cy, cx]

        # Consideriamo solo i voxel dopo il buildup (dalle profondità medie in poi)
        valid_z = pdd_ref_n > 20.0
        diff_cfm_cax = np.mean(pdd_cfm_n[valid_z] - pdd_ref_n[valid_z])
        diff_nsf_cax = np.mean(pdd_nsf_n[valid_z] - pdd_ref_n[valid_z])

        print(f"\n📁 CLASSE: {field}")
        print(f"   Voxel Dmax CAX (Z): index {z_dmax_idx}/{nz} | Dose Ref CAX: {cax_dmax_ref:.4e}")
        print(f"   Dose Max Asse / Dose Max Assoluta Ref: {(cax_dmax_ref / ref.max())*100:.1f}%")
        print(f"   --- Confronto PDD (Normalizzato su CAX Dmax Reference) ---")
        print(f"   Δ medio CFM vs Ref lungo CAX: {diff_cfm_cax:+.2f}%")
        print(f"   Δ medio NSF vs Ref lungo CAX: {diff_nsf_cax:+.2f}%")
        
        # Stampa primi punti PDD dopo dmax
        print(f"   {'Z-idx':<6} | {'Ref PDD %':<12} | {'CFM PDD %':<12} | {'NSF PDD %':<12}")
        print("   " + "-" * 50)
        for z in range(z_dmax_idx, min(z_dmax_idx + 8, nz)):
            print(f"   {z:<6} | {pdd_ref_n[z]:>10.2f}% | {pdd_cfm_n[z]:>10.2f}% | {pdd_nsf_n[z]:>10.2f}%")

if __name__ == "__main__":
    main()
