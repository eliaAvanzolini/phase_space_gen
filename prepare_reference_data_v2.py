import os
import glob
import h5py
import numpy as np
import uproot
import argparse


def load_pool(pool_base, folder, gate_branches):
    """Carica tutte le part*.root di una classe da una pool, ritorna array (N,7) o None."""
    path_pattern = f"{pool_base}/{folder}/{folder}_phsp_part*.root"
    files = sorted(glob.glob(path_pattern))
    if not files:
        return None

    pool_chunks = []
    for fpath in files:
        with uproot.open(fpath) as f:
            keys = f.keys()
            if not keys:
                continue
            main_key = keys[0]
            tree = f[main_key]
            arrays = tree.arrays(gate_branches, library="np")
            n_events = len(arrays["KineticEnergy"])

            chunk = np.zeros((n_events, 7), dtype=np.float32)
            chunk[:, 0] = arrays["PrePosition_X"] / 10.0
            chunk[:, 1] = arrays["PrePosition_Y"] / 10.0
            chunk[:, 2] = arrays["PrePosition_Z"] / 10.0
            chunk[:, 3] = arrays["PreDirection_X"]
            chunk[:, 4] = arrays["PreDirection_Y"]
            chunk[:, 5] = arrays["PreDirection_Z"]
            chunk[:, 6] = arrays["KineticEnergy"]
            pool_chunks.append(chunk)

    if not pool_chunks:
        return None
    return np.concatenate(pool_chunks, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/conditional_jaws_ref_dataset.h5",
                         help="Path del file h5 di output (reference fuso)")
    args = parser.parse_args()

    print("\n==================================================")
    print("CREAZIONE DATASET DI REFERENCE FUSO")
    print("==================================================")

    gate_configs = {
        "6mv_5x5": [6.0, 2.5, 2.5],
        "6mv_10x10": [6.0, 5.0, 5.0],
        "6mv_20x20": [6.0, 10.0, 10.0],
        "10mv_5x5": [10.0, 2.5, 2.5],
        "10mv_10x10": [10.0, 5.0, 5.0],
        "10mv_20x20": [10.0, 10.0, 10.0],
    }

    # Pool "standard": part2 (originale) + part3 + part4
    SOURCE_POOLS = [
        "outputs/gate_jaw_ref",        # part2
        "outputs/gate_jaw_ref_part3",  # part3
        "outputs/gate_jaw_ref_part4",  # part4
    ]

    # Per le classi 6mv, il gate_jaw_ref (part2) ORIGINALE risultava gonfiato
    # 4.5-7.4x rispetto a part1/part3/part4 (bug MT senza entry_start, stesso
    # difetto trovato e corretto in dose_validation, 8.09x misurato li'). E'
    # stato rigenerato correttamente in outputs/gate_jaw_ref_6mv_part2_fixed:
    # per queste 3 classi sostituiamo "outputs/gate_jaw_ref" con la versione
    # fixed, invece di usare quella originale sospetta.
    POOL_OVERRIDE_FOR_CLASS = {
        "6mv_5x5": ["outputs/gate_jaw_ref_6mv_part2_fixed",
                    "outputs/gate_jaw_ref_part3", "outputs/gate_jaw_ref_part4"],
        "6mv_10x10": ["outputs/gate_jaw_ref_6mv_part2_fixed",
                      "outputs/gate_jaw_ref_part3", "outputs/gate_jaw_ref_part4"],
        "6mv_20x20": ["outputs/gate_jaw_ref_6mv_part2_fixed",
                      "outputs/gate_jaw_ref_part3", "outputs/gate_jaw_ref_part4"],
    }

    gate_branches = [
        "PrePosition_X", "PrePosition_Y", "PrePosition_Z",
        "PreDirection_X", "PreDirection_Y", "PreDirection_Z",
        "KineticEnergy"
    ]

    with h5py.File(args.out, "w") as h5_f:
        for folder, cond_vec in gate_configs.items():
            active_pools = POOL_OVERRIDE_FOR_CLASS.get(folder, SOURCE_POOLS)
            if folder in POOL_OVERRIDE_FOR_CLASS:
                print(f"  {folder}: pool sostituiti -> {active_pools}")

            folder_ps = []
            n_per_pool = {}

            for pool_base in active_pools:
                pool_arr = load_pool(pool_base, folder, gate_branches)
                if pool_arr is None:
                    n_per_pool[pool_base] = 0
                    continue
                n_per_pool[pool_base] = len(pool_arr)
                folder_ps.append(pool_arr)

            if not folder_ps:
                print(f"  {folder}: NESSUN dato trovato in nessuna pool, salto.")
                continue

            folder_ps = np.concatenate(folder_ps, axis=0)
            grp = h5_f.create_group(folder)
            grp.create_dataset("phase_space", data=folder_ps, compression="gzip")
            grp.create_dataset("condition", data=np.array(cond_vec, dtype=np.float32))
            grp.attrs["n_particles"] = len(folder_ps)
            for pool_base, n in n_per_pool.items():
                grp.attrs[f"n_from_{os.path.basename(pool_base)}"] = n

            breakdown = " + ".join(f"{os.path.basename(k)}={v}" for k, v in n_per_pool.items() if v > 0)
            print(f"  {folder}: {len(folder_ps)} particelle totali ({breakdown})")

    print(f"\nFatto -> {args.out}")
    print("NOTA: prima di usare questo file, assicurati di aver girato")
    print("verify_and_merge_gate_outputs_v2.py --check-cross-pool sulle pool")
    print("effettivamente usate (occhio a includere gate_jaw_ref_6mv_part2_fixed")
    print("al posto di gate_jaw_ref per le classi 6mv nel check).")


if __name__ == "__main__":
    main()
