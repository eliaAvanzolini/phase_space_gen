import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).parent / "data"))
from read_iaea_phsp import read_iaea_multi  # riusa il parser IAEA gia' in uso nel progetto


def ensure_matching_headers(phsp_paths):
    """
    _find_header() in read_iaea_phsp.py cerca l'header con lo stesso nome del
    file (o con suffissi noti tipo _part1/_part2 rimossi). I file 25MV usano
    pero' la convenzione '_part1_a'/'_part1_b', non riconosciuta dal fallback
    di _find_header() (che rimuoverebbe solo '_part1', lasciando '_a'/'_b'
    residuo, senza mai trovare il match con l'header 'ELEKTA_PRECISE_25mv_part1.IAEAheader').
    Fix minimale: copiamo l'header con il nome esatto atteso per ciascun file,
    senza modificare read_iaea_phsp.py (condiviso e gia' testato altrove).
    """
    for phsp_path in phsp_paths:
        phsp_path = Path(phsp_path)
        expected_header = phsp_path.with_suffix(".IAEAheader")
        if expected_header.exists():
            continue

        # Cerca l'header "base" (stessa cartella, stesso prefisso fino a _part1)
        # es. ELEKTA_PRECISE_25mv_part1_a.IAEAphsp -> ELEKTA_PRECISE_25mv_part1.IAEAheader
        stem = phsp_path.stem  # "ELEKTA_PRECISE_25mv_part1_a"
        base_stem = stem.rsplit("_", 1)[0]  # rimuove l'ultimo "_a"/"_b" -> "ELEKTA_PRECISE_25mv_part1"
        base_header = phsp_path.parent / f"{base_stem}.IAEAheader"

        if base_header.exists():
            print(f"  [fix header] copio {base_header.name} -> {expected_header.name}")
            shutil.copy(base_header, expected_header)
        else:
            print(f"  ⚠️ ATTENZIONE: nessun header trovato ne' per {phsp_path.name} "
                  f"ne' per il prefisso base {base_stem} — la lettura probabilmente fallira'.")


def fix_split_alignment(part_a_path, part_b_path, record_size=33):
    """
    Se il file e' stato spezzato in due meta' non su un confine di record (come
    verificato per il 25MV: size_a % 33 = 17, size_b % 33 = 16), il secondo file
    va letto saltando i primi (record_size - resto) byte, altrimenti OGNI record
    risulta disallineato e la lettura scarta il 100% dei dati.
    Si perde un solo record (quello a cavallo tra i due file), trascurabile su
    decine/centinaia di milioni di particelle.
    """
    part_a_path, part_b_path = Path(part_a_path), Path(part_b_path)
    size_a = part_a_path.stat().st_size
    remainder = size_a % record_size
    if remainder == 0:
        print(f"  [align check] {part_b_path.name}: split gia' allineato al record, nessun fix necessario.")
        return str(part_b_path)

    skip_bytes = record_size - remainder
    aligned_path = part_b_path.with_name(f"{part_b_path.stem}_aligned{part_b_path.suffix}")

    if aligned_path.exists():
        print(f"  [align fix] {aligned_path.name} gia' presente, riuso senza rigenerare.")
    else:
        print(f"  [align fix] {part_b_path.name}: split non allineato (resto={remainder} byte in part_a), "
              f"salto i primi {skip_bytes} byte e scrivo {aligned_path.name}")
        chunk_bytes = 64 * 1024 * 1024
        with open(part_b_path, "rb") as fin, open(aligned_path, "wb") as fout:
            fin.seek(skip_bytes)
            while True:
                buf = fin.read(chunk_bytes)
                if not buf:
                    break
                fout.write(buf)

    # L'header va copiato anche per il file "aligned" (stesso header base di part_b)
    b_header = part_b_path.with_suffix(".IAEAheader")
    aligned_header = aligned_path.with_suffix(".IAEAheader")
    if b_header.exists() and not aligned_header.exists():
        shutil.copy(b_header, aligned_header)

    return str(aligned_path)


def load_root_open_beam(root_path: str) -> np.ndarray:
    """Carica un fascio aperto gia' convertito in ROOT (schema a 7 colonne
    X,Y,Z,dX,dY,dZ,E). ATTENZIONE: questi file (creati da convert_part1_to_root.py
    / convert_iaea_generic.py a partire da read_iaea_phsp.py) hanno le posizioni
    GIA' in cm (read_iaea_phsp.py restituisce esplicitamente x_cm/y_cm/z_cm) -
    NESSUNA conversione di unita' va applicata qui. Il /10.0 (mm->cm) serve solo
    per i ROOT prodotti da simulazioni GATE (dove Geant4 salva in mm), non per
    questi file aperti convertiti direttamente da IAEA."""
    with uproot.open(root_path) as f:
        keys = f.keys()
        tree = f[keys[0]]
        arrays = tree.arrays(
            ["PrePosition_X", "PrePosition_Y", "PrePosition_Z",
             "PreDirection_X", "PreDirection_Y", "PreDirection_Z",
             "KineticEnergy"],
            library="np",
        )
    n = len(arrays["KineticEnergy"])
    ps = np.zeros((n, 7), dtype=np.float32)
    ps[:, 0] = arrays["PrePosition_X"]  # gia' in cm, NESSUNA divisione
    ps[:, 1] = arrays["PrePosition_Y"]
    ps[:, 2] = arrays["PrePosition_Z"]
    ps[:, 3] = arrays["PreDirection_X"]
    ps[:, 4] = arrays["PreDirection_Y"]
    ps[:, 5] = arrays["PreDirection_Z"]
    ps[:, 6] = arrays["KineticEnergy"]
    return ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_6mv", default="data/ELEKTA_PRECISE_6mv_part1.root",
                     help="Fascio aperto 6MV gia' convertito in ROOT (fonte training)")
    ap.add_argument("--iaea_25mv", nargs="+",
                     default=["data/ELEKTA_PRECISE_25mv_part1_a.IAEAphsp",
                              "data/ELEKTA_PRECISE_25mv_part1_b.IAEAphsp"],
                     help="File IAEA grezzi del 25MV da concatenare (fonte training)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/energy_only_train_dataset.h5")
    ap.add_argument("--target_per_class", type=int, default=80_000_000,
                     help="Cap massimo di particelle per classe (default 40M, "
                          "dimensionato per far stare 200 epoche piene di GAN "
                          "e NSF dentro le 48h di commonGPUQ, mantenendo dati "
                          "ed epoche identici tra i tre modelli per un confronto equo)")
    args = ap.parse_args()

    print("=" * 60)
    print("COSTRUZIONE DATASET DI TRAINING A SOLA ENERGIA (6MV + 25MV)")
    print("=" * 60)

    print("\nCaricamento 6MV (ROOT, fascio aperto)...")
    ps_6mv = load_root_open_beam(args.root_6mv)
    print(f"  {len(ps_6mv):,} particelle")

    print("\nCaricamento 25MV (IAEA grezzo, part1_a + part1_b)...")
    ensure_matching_headers(args.iaea_25mv)
    if len(args.iaea_25mv) == 2:
        args.iaea_25mv[1] = fix_split_alignment(args.iaea_25mv[0], args.iaea_25mv[1])
    ps_25mv = read_iaea_multi(args.iaea_25mv, remove_511=True)
    print(f"  {len(ps_25mv):,} particelle")

    # ── Bilanciamento: sottocampiona la classe piu' numerosa al conteggio
    # della piu' piccola. Qui il costo e' minimo (le due energie sono gia'
    # dello stesso ordine di grandezza), a differenza dei dataset jaws-condizionati.
    np.random.seed(args.seed)
    min_count = min(len(ps_6mv), len(ps_25mv), args.target_per_class)
    print(f"\nBilanciamento a {min_count:,} particelle per classe "
          f"(perdita: 6MV {100*(1-min_count/len(ps_6mv)):.2f}%, "
          f"25MV {100*(1-min_count/len(ps_25mv)):.2f}%)")

    if len(ps_6mv) > min_count:
        idx = np.random.choice(len(ps_6mv), size=min_count, replace=False)
        ps_6mv = ps_6mv[idx]
    if len(ps_25mv) > min_count:
        idx = np.random.choice(len(ps_25mv), size=min_count, replace=False)
        ps_25mv = ps_25mv[idx]

    phase_space = np.concatenate([ps_6mv, ps_25mv], axis=0)
    # Conditions: SOLA energia (cond_dim=1), niente jaw per questo esperimento
    conditions = np.concatenate([
        np.full((min_count, 1), 6.0, dtype=np.float32),
        np.full((min_count, 1), 25.0, dtype=np.float32),
    ], axis=0)

    # Shuffle finale (altrimenti il dataset e' ordinato a blocchi per energia,
    # e col DataLoader shuffle=True in train.py non e' un problema, ma meglio
    # non fare affidamento solo su quello)
    perm = np.random.permutation(len(phase_space))
    phase_space = phase_space[perm]
    conditions = conditions[perm]

    print(f"\nDataset finale: {len(phase_space):,} particelle totali "
          f"({min_count:,} per classe x 2 classi)")

    with h5py.File(args.out, "w") as f:
        f.create_dataset("phase_space", data=phase_space, compression="gzip")
        f.create_dataset("conditions", data=conditions, compression="gzip")
        f.attrs["balanced"] = True
        f.attrs["n_total"] = len(phase_space)
        f.attrs["n_per_class"] = min_count
        f.attrs["classes"] = "6MV, 25MV (fasci aperti, no jaw)"
        f.attrs["cond_dim"] = 1
        f.attrs["note"] = ("10MV escluso di proposito: riservato come held-out "
                            "per la valutazione dell'interpolazione")

    print(f"\nSalvato: {args.out}")
    print("cond_dim=1 (sola energia) — verificare che train.py/models gestiscano "
          "correttamente cond_dim=1 (dovrebbe essere automatico dato che cond_dim "
          "viene dedotto da conditions.shape[1] in prepare_data())")


if __name__ == "__main__":
    main()
