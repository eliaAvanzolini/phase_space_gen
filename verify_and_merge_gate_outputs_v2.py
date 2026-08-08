#!/usr/bin/env python3
"""
verify_and_merge_gate_outputs_v2.py
===================================
1. Verifica conteggi particelle tra part2, part3, part4 (part2_fixed per 6MV)
2. Verifica assenza di sovrapposizioni/duplicati cross-pool (arrotondamento a 5 decimali)
"""

import argparse
import glob
import os
import numpy as np
import uproot

KINEMATIC_BRANCHES = [
    "KineticEnergy",
    "PrePosition_X",
    "PrePosition_Y",
    "PrePosition_Z",
    "PreDirection_X",
    "PreDirection_Y",
    "PreDirection_Z",
]
DECIMALS = 5

CLASSES = [
    "6mv_5x5",
    "6mv_10x10",
    "6mv_20x20",
    "10mv_5x5",
    "10mv_10x10",
    "10mv_20x20",
]


def get_pool_path(pool_name, classname):
    """Restituisce il percorso corretto per part2 (fixed per 6MV, std per 10MV) e part3/part4."""
    if pool_name == "part2":
        if classname.startswith("6mv"):
            return "outputs/gate_jaw_ref_6mv_part2_fixed"
        else:
            return "outputs/gate_jaw_ref"
    elif pool_name == "part3":
        return "outputs/gate_jaw_ref_part3"
    elif pool_name == "part4":
        return "outputs/gate_jaw_ref_part4"
    raise ValueError(f"Pool non valida: {pool_name}")


def load_class_keys(base_dir, classname):
    pattern = os.path.join(base_dir, classname, f"{classname}_phsp_part*.root")
    files = sorted(glob.glob(pattern))
    if not files:
        return None, 0

    all_keys = set()
    n_total = 0
    for f in files:
        with uproot.open(f) as fh:
            keys = [k.split(";")[0] for k in fh.keys()]
            tree_name = next(k for k in keys if hasattr(fh[k], "num_entries"))
            arr = fh[tree_name].arrays(KINEMATIC_BRANCHES, library="np")
        stacked = np.round(
            np.column_stack([arr[b] for b in KINEMATIC_BRANCHES]), DECIMALS
        )
        n_total += len(stacked)
        for row in stacked:
            all_keys.add(tuple(row))
    return all_keys, n_total


def check_cross_pool():
    print("=" * 90)
    print("CHECK CROSS-POOL E VERIFICA STATISTICHE (part2_fixed / part3 / part4)")
    print("=" * 90)

    pools = ["part2", "part3", "part4"]

    for classname in CLASSES:
        print(f"\n📁 CLASSE: {classname}")
        loaded = {}
        for pool_name in pools:
            base_dir = get_pool_path(pool_name, classname)
            if not os.path.isdir(os.path.join(base_dir, classname)):
                print(
                    f"  [{pool_name}] Cartella non trovata: {base_dir}/{classname}"
                )
                continue

            keys, n_total = load_class_keys(base_dir, classname)
            if keys is None:
                print(f"  [{pool_name}] Nessun file ROOT trovato in {base_dir}")
                continue

            loaded[pool_name] = keys
            print(
                f"  [{pool_name}] Total: {n_total:,} particelle | Uniche: {len(keys):,}"
            )

        # Confronto cross-pool
        names = list(loaded.keys())
        any_overlap = False
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                overlap = loaded[a] & loaded[b]
                if overlap:
                    any_overlap = True
                    print(
                        f"  ⚠️ ATTENZIONE: {len(overlap)} particelle in comune tra {a} e {b}!"
                    )
                else:
                    print(f"  ✅ Nessuna intersezione tra {a} e {b}")

        if not any_overlap and len(names) > 1:
            print(
                f"  => {classname}: OK, tutte le pool sono disgiunte e uniche!"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-cross-pool", action="store_true")
    args = ap.parse_args()
    if args.check_cross_pool:
        check_cross_pool()
    else:
        print("Specificare --check-cross-pool per eseguire il controllo.")


if __name__ == "__main__":
    main()
