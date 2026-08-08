import sys
import gc
import numpy as np
import uproot
from pathlib import Path
from itertools import combinations

gate_branches = [
    "PrePosition_X",
    "PrePosition_Y",
    "PrePosition_Z",
    "PreDirection_X",
    "PreDirection_Y",
    "PreDirection_Z",
    "KineticEnergy",
]

parts = ["part1", "part2", "part3", "part4"]
energies = ["6mv", "10mv"]
data_dir = Path("data")

def check_pair(energy: str, p1: str, p2: str):
    # Salta il confronto part1 vs part2 (già verificato con successo)
    if p1 == "part1" and p2 == "part2":
        print(f"⏩ [{energy.upper()}] Skip part1 vs part2 (già verificato in precedenza).", flush=True)
        return

    f1_path = data_dir / f"ELEKTA_PRECISE_{energy}_{p1}.root"
    f2_path = data_dir / f"ELEKTA_PRECISE_{energy}_{p2}.root"

    if not f1_path.exists() or not f2_path.exists():
        print(f"⚠️ File mancanti: {f1_path.name} o {f2_path.name}", flush=True)
        return

    print(f"\n🔍 [{energy.upper()}] Caricamento e confronto {p1} vs {p2}...", flush=True)

    with uproot.open(f1_path) as f1:
        t1 = f1[f1.keys()[0]]
        arr1 = t1.arrays(gate_branches, library="np")
        data1 = np.column_stack([arr1[b] for b in gate_branches])

    with uproot.open(f2_path) as f2:
        t2 = f2[f2.keys()[0]]
        arr2 = t2.arrays(gate_branches, library="np")
        data2 = np.column_stack([arr2[b] for b in gate_branches])

    print(f"   Analisi intersezione su {len(data1):,} vs {len(data2):,} particelle...", flush=True)

    v1 = np.ascontiguousarray(data1.round(5)).view(
        np.dtype((np.void, data1.dtype.itemsize * data1.shape[1]))
    )
    v2 = np.ascontiguousarray(data2.round(5)).view(
        np.dtype((np.void, data2.dtype.itemsize * data2.shape[1]))
    )

    duplicati = np.intersect1d(v1, v2)
    n_dup = len(duplicati)

    if n_dup == 0:
        print(f"  ✅ ZERO duplicati trovati tra {p1} e {p2}!", flush=True)
    else:
        pct = (n_dup / min(len(data1), len(data2))) * 100
        print(f"  ⚠️ ATTENZIONE: Trovati {n_dup:,} duplicati ({pct:.4f}%)!", flush=True)

    del data1, data2, v1, v2, duplicati
    gc.collect()

if __name__ == "__main__":
    print("🚀 Inizio verifica incrociata Anti-Overlap (6MV & 10MV)...", flush=True)
    for energy in energies:
        print(f"\n==================== ENERGIA {energy.upper()} ====================", flush=True)
        for p1, p2 in combinations(parts, 2):
            check_pair(energy, p1, p2)
    print("\n🎉 Controllo Anti-Overlap completato con successo!", flush=True)
