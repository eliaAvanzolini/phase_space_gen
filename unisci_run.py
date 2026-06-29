import os
import SimpleITK as sitk

base_dir = "outputs/dose_validation"
models = ["cfm", "nsf", "gan", "reference"]

print("\n=============================================================")
print(" 🔮 INIZIO FUSIONE DELLE RUN PARALLELE (100M STORIE TOTALI)")
print("=============================================================\n")

for model in models:
    merged_img = None
    count = 0
    print(f"📦 Elaborazione modello: {model.upper()}")
    
    for run in range(1, 11):
        file_path = os.path.join(base_dir, f"run_{run}", f"dose_{model}_dose.mhd")
        
        if os.path.exists(file_path):
            img = sitk.ReadImage(file_path)
            if merged_img is None:
                merged_img = img
            else:
                merged_img = merged_img + img
            count += 1
        else:
            print(f"  ⚠️ Run {run} non trovata per {model}")
            
    if merged_img is not None:
        output_path = os.path.join(base_dir, f"dose_{model}_dose.mhd")
        sitk.WriteImage(merged_img, output_path)
        print(f"  ✓ Unito con successo ({count}/10 run) -> {output_path}\n")

print("=============================================================")
print(" 🎉 FUSIONE COMPLETATA! I file da 100M sono pronti.")
print("=============================================================\n")
