import uproot
import numpy as np

print("\n=======================================================")
print(" 🔬 VERIFICA DIRETTI DEI DIZIONARI DELLE PARTICELLE")
print("=======================================================\n")

for name in ["reference", "cfm", "nsf", "gan"]:
    path = f"outputs/outputs_condizionali_6mv_5x5/run_1/gen_{name}.root"
    try:
        with uproot.open(path) as f:
            tree = f["PhaseSpace"]
            keys = ["X", "Y", "Z", "dX", "dY", "dZ", "E"]
            data = tree.arrays(keys, entry_stop=1)
            
            print(f"🧠 MODELLO: {name.upper()}")
            print(f"  · Numero particelle nel file: {tree.num_entries}")
            for k in keys:
                print(f"    - {k}: {data[k][0]:.4f}")
            print("-" * 40)
    except Exception as e:
        print(f"❌ Errore di lettura per {name.upper()}: {e}\n")
