import os
import SimpleITK as sitk
import numpy as np

BASE_DIR = "outputs/dose_validation"
FIELDS = ["6mv_5x5", "6mv_10x10", "6mv_20x20", "10mv_5x5", "10mv_10x10", "10mv_20x20"]
TAGS = ["reference", "cfm", "nsf"]

def main():
    print("=" * 95)
    print("VERIFICA SPOTS DI RUMORE E OUTLIER SU D_MAX")
    print("=" * 95)

    for field in FIELDS:
        print(f"\n📁 CLASSE: {field}")
        print("-" * 95)
        print(f"{'Mappa':<12} | {'Dmax (1°)':<12} | {'2° Max':<12} | {'Ratio 2°/1°':<12} | {'Mean Top-10 / Max':<18} | {'P99.9 / Max':<12}")
        print("-" * 95)
        
        for tag in TAGS:
            path = os.path.join(BASE_DIR, field, f"dose_{tag}_dose.mhd")
            if not os.path.exists(path):
                print(f"{tag:<12} | FILE NON TROVATO in {path}")
                continue
                
            img = sitk.ReadImage(path)
            arr = sitk.GetArrayFromImage(img).astype(np.float64)
            
            if arr.max() == 0:
                print(f"{tag:<12} | Mappa vuota (tutti zeri)")
                continue

            top10 = np.sort(arr.flatten())[-10:]
            dmax = arr.max()
            second_max = top10[-2]
            
            ratio_2nd = (second_max / dmax) * 100.0
            ratio_top10 = (top10.mean() / dmax) * 100.0
            
            # Percentile 99.9 escludendo i voxel a zero
            nonzero = arr[arr > 0]
            p99_9 = np.percentile(nonzero, 99.9) if len(nonzero) > 0 else 0
            ratio_p99_9 = (p99_9 / dmax) * 100.0
            
            print(f"{tag:<12} | {dmax:<12.4e} | {second_max:<12.4e} | {ratio_2nd:>10.2f}% | {ratio_top10:>16.2f}% | {ratio_p99_9:>10.2f}%")

if __name__ == "__main__":
    main()
