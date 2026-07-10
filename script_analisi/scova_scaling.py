import torch
import json
from pathlib import Path

print("\n=======================================================")
print(" 🔬 ISPEZIONE DEI PARAMETRI DI SCALING DELLA CONDIZIONE")
print("=======================================================\n")

# 1. Controlliamo se nel checkpoint del CFM ci sono costanti di scaling per le condizioni
ckpt_path = "outputs/cfm_conditional_6mv_10mv/best_model.pt"
if Path(ckpt_path).exists():
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print("🧠 CHECKPOINT CFM KEYS:")
        print(f"  · Chiavi disponibili: {list(ckpt.keys())}")
        if "cond_stats" in ckpt:
            print(f"  · 🎯 Trovate statistiche condizione nel checkpoint: {ckpt['cond_stats']}")
        elif "stats" in ckpt:
            print(f"  · 🎯 Statistiche generali nel checkpoint: {ckpt['stats']}")
    except Exception as e:
        print(f"  ⚠️ Impossibile leggere il checkpoint: {e}")
print("-" * 50)

# 2. Greppiamo i file di training per vedere se c'è una normalizzazione manuale delle condizioni
print("📂 ANALISI DEI CODICI DI ADDESTRAMENTO (RICERCA SCALING):")
for script in ["train.py", "prepare_conditional_data.py", "data/synthetic_linac.py"]:
    if Path(script).exists():
        print(f"\n  🔍 Spio dentro: {script}")
        with open(script, "r") as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            if any(k in line.lower() for k in ["cond", "condition"]) and any(op in line for op in ["/", "-", "*", "min", "max", "std", "mu"]):
                print(f"    [Linea {idx+1}]: {line.strip()}")
print("=======================================================\n")
