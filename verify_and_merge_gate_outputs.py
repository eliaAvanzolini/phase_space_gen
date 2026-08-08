#!/usr/bin/env python3
"""
Verifica integrita', verifica duplicati e merge dei file GATE jaw-collimated
suddivisi in parti (part1..partN) per ciascuna classe.

Uso:
    python verify_and_merge_gate_outputs.py --check
    python verify_and_merge_gate_outputs.py --check --classdir outputs/gate_jaw/6mv_5x5
    python verify_and_merge_gate_outputs.py --merge

Il check va SEMPRE lanciato ed esaminato prima del merge. Il merge si rifiuta
di procedere per una classe se il check su quella classe ha trovato problemi
(a meno di --force).
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import uproot

# Branch attesi in output dal PhaseSpaceActor (attributes = KineticEnergy,
# PrePosition, PreDirection, ParticleName)
KINEMATIC_BRANCHES = [
    "KineticEnergy",
    "PrePosition_X", "PrePosition_Y", "PrePosition_Z",
    "PreDirection_X", "PreDirection_Y", "PreDirection_Z",
]
DECIMALS = 5  # stessa precisione usata per il check part1/part2

BASE_DIRS = ["outputs/gate_jaw", "outputs/gate_jaw_ref"]


def find_tree_name(path):
    with uproot.open(path) as f:
        keys = [k.split(";")[0] for k in f.keys()]
        # Prendiamo il primo TTree trovato
        for k in keys:
            obj = f[k]
            if hasattr(obj, "num_entries"):
                return k
    raise RuntimeError(f"Nessun TTree trovato in {path}")


def check_integrity(path):
    """Ritorna (ok, n_entries, msg)"""
    try:
        tree_name = find_tree_name(path)
        with uproot.open(path) as f:
            tree = f[tree_name]
            n = tree.num_entries
            missing = [b for b in KINEMATIC_BRANCHES if b not in tree.keys()]
            if missing:
                return False, n, f"branch mancanti: {missing}"
            if n == 0:
                return False, n, "file con 0 entries"
            # forza la lettura effettiva di un chunk per scovare file troncati
            _ = tree.arrays(KINEMATIC_BRANCHES, entry_start=0,
                             entry_stop=min(1000, n), library="np")
            return True, n, "ok"
    except Exception as e:
        return False, -1, f"ERRORE APERTURA: {e}"


def list_class_dirs(only=None):
    if only:
        return [only]
    dirs = []
    for base in BASE_DIRS:
        if not os.path.isdir(base):
            continue
        for d in sorted(os.listdir(base)):
            full = os.path.join(base, d)
            if os.path.isdir(full):
                dirs.append(full)
    return dirs


def parts_in_class(classdir):
    """Ritorna lista (path, part_index) ordinata. Segnala nomi anomali."""
    pattern = os.path.join(classdir, "*_phsp_part*.root")
    files = glob.glob(pattern)
    parsed = []
    anomalies = []
    for f in files:
        m = re.search(r"_phsp_part(\d+)\.root$", os.path.basename(f))
        if not m:
            anomalies.append(f)
            continue
        parsed.append((f, int(m.group(1))))
    parsed.sort(key=lambda x: x[1])
    return parsed, anomalies


def check_class(classdir):
    print(f"\n=== {classdir} ===")
    parts, anomalies = parts_in_class(classdir)

    if anomalies:
        print(f"  ATTENZIONE: {len(anomalies)} file con nome non standard (non _phsp_partN.root):")
        for a in anomalies:
            print(f"    - {a}")

    # Rileva indici duplicati (es. due file per part5)
    idx_seen = {}
    for f, idx in parts:
        idx_seen.setdefault(idx, []).append(f)
    dup_idx = {i: fl for i, fl in idx_seen.items() if len(fl) > 1}
    if dup_idx:
        print(f"  ATTENZIONE: indici di parte duplicati:")
        for i, fl in dup_idx.items():
            for f in fl:
                sz = os.path.getsize(f)
                print(f"    part{i}: {f} ({sz/1e6:.1f} MB)")

    # Integrita' di ognuno
    all_ok = True
    total_entries = 0
    arrays_per_file = {}
    for f, idx in parts:
        ok, n, msg = check_integrity(f)
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] part{idx:03d}: {n:>10} entries  ({msg})  {f}")
        if not ok:
            all_ok = False
        else:
            total_entries += n

    print(f"  Totale entries (file integri): {total_entries}")

    # Check duplicati TRA le parti (stesso metodo a 5 decimali di part1/part2)
    print("  Controllo duplicati tra le parti (arrotondato a {} decimali)...".format(DECIMALS))
    seen_keys = set()
    n_dupes = 0
    for f, idx in parts:
        ok, n, msg = check_integrity(f)
        if not ok:
            continue
        with uproot.open(f) as fh:
            tree_name = find_tree_name(f)
            arr = fh[tree_name].arrays(KINEMATIC_BRANCHES, library="np")
        stacked = np.round(
            np.column_stack([arr[b] for b in KINEMATIC_BRANCHES]), DECIMALS
        )
        # chiave hashabile per riga
        keys = [tuple(row) for row in stacked]
        before = len(seen_keys)
        for k in keys:
            if k in seen_keys:
                n_dupes += 1
            else:
                seen_keys.add(k)
        # free memory
        del arr, stacked

    if n_dupes > 0:
        print(f"  ATTENZIONE: {n_dupes} vettori duplicati trovati tra le parti!")
        all_ok = False
    else:
        print("  Nessun duplicato trovato tra le parti.")

    return all_ok and not anomalies and not dup_idx, total_entries


def merge_class(classdir, out_name=None, force=False):
    parts, anomalies = parts_in_class(classdir)
    if anomalies or any(len(v) > 1 for v in
                         {i: [f for f, ii in parts if ii == i] for _, i in parts}.values()):
        if not force:
            print(f"  SKIP merge per {classdir}: anomalie rilevate, lanciare --check e risolvere prima "
                  f"(o usare --force per ignorare)")
            return

    if out_name is None:
        out_name = os.path.join(classdir, os.path.basename(classdir) + "_phsp_merged.root")

    print(f"  Merging {len(parts)} file -> {out_name}")
    first = True
    with uproot.recreate(out_name) as fout:
        for f, idx in parts:
            tree_name = find_tree_name(f)
            with uproot.open(f) as fin:
                tree = fin[tree_name]
                n = tree.num_entries
                batch = 2_000_000
                for start in range(0, n, batch):
                    stop = min(start + batch, n)
                    arrays = tree.arrays(entry_start=start, entry_stop=stop, library="np")
                    if first:
                        fout["PhaseSpace"] = arrays
                        first = False
                    else:
                        fout["PhaseSpace"].extend(arrays)
    print(f"  Merge completato: {out_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--classdir", default=None, help="Limita a una sola classe (path)")
    ap.add_argument("--force", action="store_true", help="Forza il merge anche con anomalie")
    args = ap.parse_args()

    if not args.check and not args.merge:
        print("Specificare --check e/o --merge")
        sys.exit(1)

    class_dirs = list_class_dirs(args.classdir)
    results = {}

    if args.check:
        for cd in class_dirs:
            ok, total = check_class(cd)
            results[cd] = ok
        print("\n=== RIEPILOGO CHECK ===")
        for cd, ok in results.items():
            print(f"  {'OK ' if ok else 'FAIL'}  {cd}")

    if args.merge:
        for cd in class_dirs:
            ok = results.get(cd, True)  # se non e' stato fatto check, assume ok (rischioso: consigliato farlo sempre insieme)
            if not ok and not args.force:
                print(f"SKIP merge {cd}: check fallito")
                continue
            merge_class(cd, force=args.force)


if __name__ == "__main__":
    main()
