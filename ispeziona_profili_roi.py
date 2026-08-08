import os
import SimpleITK as sitk
import numpy as np

BASE_DIR = "outputs/dose_validation"
FIELDS = ["6mv_5x5", "6mv_10x10", "6mv_20x20", "10mv_5x5", "10mv_10x10", "10mv_20x20"]
ROI_RADIUS = 2  # Finestra (2*2+1)x(2*2+1) = 5x5 voxel (~1x1 cm^2) attorno al centro

def main():
    print("=" * 90)
    print("PROFILI PDD CON INTEGRAZIONE SPAZIALE ROI (5x5 voxel attorno a CAX)")
    print("=" * 90)

    for field in FIELDS:
        ref_path = os.path.join(BASE_DIR, field, "dose_reference_dose.mhd")
        cfm_path = os.path.join(BASE_DIR, field, "dose_cfm_dose.mhd")
        nsf_path = os.path.join(BASE_DIR, field, "dose_nsf_dose.mhd")

        if not os.path.exists(ref_path):
            continue

        ref = sitk.GetArrayFromImage(sitk.ReadImage(ref_path)).astype(np.float64)
        cfm = sitk.GetArrayFromImage(sitk.ReadImage(cfm_path)).astype(np.float64)
        nsf = sitk.GetArrayFromImage(sitk.ReadImage(nsf_path)).astype(np.float64)

        nz, ny, nx = ref.shape
        cy, cx = ny // 2, nx // 2

        # Media spaziale 2D attorno all'asse centrale per ogni fetta Z
        pdd_ref = ref[:, cy-ROI_RADIUS:cy+ROI_RADIUS+1, cx-ROI_RADIUS:cx+ROI_RADIUS+1].mean(axis=(1, 2))
        pdd_cfm = cfm[:, cy-ROI_RADIUS:cy+ROI_RADIUS+1, cx-ROI_RADIUS:cx+ROI_RADIUS+1].mean(axis=(1, 2))
        pdd_nsf = nsf[:, cy-ROI_RADIUS:cy+ROI_RADIUS+1, cx-ROI_RADIUS:cx+ROI_RADIUS+1].mean(axis=(1, 2))

        # Normalizziamo sulla Dmax integrata della Reference
        cax_dmax_ref = pdd_ref.max()
        if cax_dmax_ref == 0:
            continue

        pdd_ref_n = (pdd_ref / cax_dmax_ref) * 100.0
        pdd_cfm_n = (pdd_cfm / cax_dmax_ref) * 100.0
        pdd_nsf_n = (pdd_nsf / cax_dmax_ref) * 100.0

        z_max = np.argmax(pdd_ref)

        print(f"\n📁 CLASSE: {field} (Dmax integrata Ref @ Z-idx {z_max})")
        print(f"   {'Z-idx':<6} | {'Ref PDD %':<12} | {'CFM PDD %':<12} | {'NSF PDD %':<12}")
        print("   " + "-" * 50)
        for z in range(max(0, z_max - 2), min(nz, z_max + 8)):
            print(f"   {z:<6} | {pdd_ref_n[z]:>10.2f}% | {pdd_cfm_n[z]:>10.2f}% | {pdd_nsf_n[z]:>10.2f}%")

if __name__ == "__main__":
    main()
