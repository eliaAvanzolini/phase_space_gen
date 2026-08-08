import os
import SimpleITK as sitk
import numpy as np

BASE_DIR = "outputs/dose_validation"
FIELDS = ["6mv_5x5", "6mv_10x10", "6mv_20x20", "10mv_5x5", "10mv_10x10", "10mv_20x20"]
TAGS = ["reference", "cfm", "nsf"]


def main():
    print("=" * 105)
    print("VERIFICA INCERTEZZA STATISTICA AL VOXEL DI DOSE MASSIMA")
    print("=" * 105)

    for field in FIELDS:
        print(f"\n📁 CLASSE: {field}")
        print("-" * 105)
        print(f"{'Mappa':<12} | {'Dmax':<12} | {'Uncert @ Dmax':<16} | {'Uncert Mediana (>50% Dmax)':<28} | Status")
        print("-" * 105)

        for tag in TAGS:
            dose_path = os.path.join(BASE_DIR, field, f"dose_{tag}_dose.mhd")
            unc_path = os.path.join(BASE_DIR, field, f"dose_{tag}_dose_uncertainty.mhd")

            if not (os.path.exists(dose_path) and os.path.exists(unc_path)):
                print(f"{tag:<12} | File non trovati")
                continue

            dose = sitk.GetArrayFromImage(sitk.ReadImage(dose_path)).astype(np.float64)
            unc = sitk.GetArrayFromImage(sitk.ReadImage(unc_path)).astype(np.float64)

            idx_max = np.unravel_index(np.argmax(dose), dose.shape)
            unc_at_max = unc[idx_max] * 100.0  # Converti in %

            high_dose_mask = dose > 0.5 * dose.max()
            unc_median_highdose = np.median(unc[high_dose_mask]) * 100.0

            flag = "⚠️ RUMORE STATISTICO" if unc_at_max > 2.0 * unc_median_highdose else "OK (Segnale)"
            print(f"{tag:<12} | {dose.max():<12.4e} | {unc_at_max:>14.2f}% | {unc_median_highdose:>26.2f}% | {flag}")


if __name__ == "__main__":
    main()
