import os
import sys
import glob
import uproot
import numpy as np
from pathlib import Path

def load_raw_gate_pool():
    files = sorted(glob.glob("outputs/gate_jaw/6mv_5x5/6mv_5x5_phsp_part*.root"))
    if not files:
        print("❌ Errore: File ROOT primari non trovati!")
        sys.exit(1)
        
    gate_branches = [
        "PrePosition_X", "PrePosition_Y", "PrePosition_Z",
        "PreDirection_X", "PreDirection_Y", "PreDirection_Z",
        "KineticEnergy"
    ]
    
    chunks = []
    for fpath in files:
        with uproot.open(fpath) as f:
            if not f.keys(): continue
            tree = f[f.keys()[0]]
            arrays = tree.arrays(gate_branches, library="np")
            chunk = np.zeros((len(arrays["KineticEnergy"]), 7), dtype=np.float32)
            chunk[:, 0] = arrays["PrePosition_X"] / 10.0
            chunk[:, 1] = arrays["PrePosition_Y"] / 10.0
            chunk[:, 2] = arrays["PrePosition_Z"] / 10.0
            chunk[:, 3] = arrays["PreDirection_X"]
            chunk[:, 4] = arrays["PreDirection_Y"]
            chunk[:, 5] = arrays["PreDirection_Z"]
            chunk[:, 6] = arrays["KineticEnergy"]
            chunks.append(chunk)
    return np.concatenate(chunks, axis=0)

print("=========================================================")
print("🔬 ANALISI DI UNICITÀ BIT-BY-BIT SUL POOL RAW DI GATE")
print("=========================================================")

raw_pool = load_raw_gate_pool()
total_rows = len(raw_pool)

print(f"📦 Matrice RAW caricata: {total_rows} righe totali.")
print("⏳ Calcolo dei duplicati analitici in corso...")

# Estrazione delle righe uniche e dei loro conteggi su tutte e 7 le colonne
unique_rows, counts = np.unique(raw_pool, axis=0, return_counts=True)

print("\n📊 -----------------------------------------------------")
print("📊 VERDIETTO RILEVATO SUI DATI COMPILATI:")
print("-----------------------------------------------------")
print(f" Righe uniche effettive:     {len(unique_rows)} su {total_rows}")
print(f" Percentuale di unicità:     {(len(unique_rows)/total_rows)*100:.2f}%")
print(f" Duplicazione max trovata:   {counts.max()} copie identiche di una riga")
print(f" Righe con più di 1 copia:   {(counts > 1).sum()}")
print("-----------------------------------------------------")

print("\n⚙️ IMPLICAZIONE METODOLOGICA:")
if len(unique_rows) < total_rows:
    print(" ⚠️ ATTENZIONE: Data Leakage confermato. Lo split casuale per righe è contaminato.")
    print(" 👉 Azione: È obbligatorio passare a uno split basato su file (Held-Out File Split).")
else:
    print(" ✅ Pool pulito. Nessun duplicato bit-per-bit rilevato.")
print("=========================================================")
