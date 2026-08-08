import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import uproot


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


def quick_duplicate_spotcheck(parts: dict, n_sample=200_000, decimals=5, seed=0):
    """
    Controllo rapido e approssimato (non esaustivo - impraticabile a piena
    scala su 4 file da ~125M righe ciascuno): sottocampiona n_sample vettori
    da ciascun part e cerca corrispondenze esatte (a 5 decimali) tra le coppie.
    Serve solo da rete di sicurezza veloce, non sostituisce un check completo
    come quello fatto in precedenza su part1/part2.
    """
    rng = np.random.default_rng(seed)
    samples = {}
    for name, ps in parts.items():
        idx = rng.choice(len(ps), size=min(n_sample, len(ps)), replace=False)
        rounded = np.round(ps[idx], decimals)
        samples[name] = set(map(tuple, rounded))

    names = list(samples.keys())
    any_overlap = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = samples[names[i]] & samples[names[j]]
            if overlap:
                any_overlap = True
                print(f"  ⚠️ ATTENZIONE: {len(overlap)} vettori in comune (su campione di "
                      f"{n_sample:,}) tra {names[i]} e {names[j]}")
            else:
                print(f"  ✅ Nessuna corrispondenza tra {names[i]} e {names[j]} "
                      f"(campione di {n_sample:,} ciascuno)")
    if not any_overlap:
        print("  => Spot-check pulito. Non e' una garanzia esaustiva, ma nessun segnale di problemi.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=[
        # NOTA: ELEKTA_PRECISE_10mv_part1.root ESCLUSO deliberatamente. Verificato
        # (check_z_per_part_10mv.py) che ha Z=0.0 cm, mentre part2/3/4 hanno tutti
        # Z=27.21 cm - piano di registrazione fisicamente diverso (probabilmente
        # all'uscita del target invece che 27.21cm a valle). Mescolarlo con gli
        # altri senza una correzione di trasporto crea una distribuzione Z
        # bimodale spuria nel reference. Restano comunque ~372M particelle da
        # part2+3+4, piu' che sufficienti.
        "data/ELEKTA_PRECISE_10mv_part2.root",
        "data/ELEKTA_PRECISE_10mv_part3.root",
        "data/ELEKTA_PRECISE_10mv_part4.root",
    ])
    ap.add_argument("--out", default="data/energy_only_10mv_reference.h5")
    ap.add_argument("--skip_spotcheck", action="store_true",
                     help="Salta il controllo rapido di duplicati tra i part")
    args = ap.parse_args()

    print("=" * 60)
    print("COSTRUZIONE REFERENCE 10MV NASCOSTO (fascio aperto, tutti i part)")
    print("=" * 60)
    print("\nNOTA: nessuno di questi part e' mai stato usato in training in "
          "questo esperimento (a sola energia, 6MV+25MV) - qui possiamo "
          "usare TUTTA la statistica disponibile senza alcun compromesso.")

    parts = {}
    for p in args.parts:
        name = Path(p).stem
        print(f"\nCaricamento {name}...")
        ps = load_root_open_beam(p)
        print(f"  {len(ps):,} particelle")
        parts[name] = ps

    if not args.skip_spotcheck:
        print(f"\n{'-'*60}\nSpot-check rapido di duplicati tra i part (approssimato)\n{'-'*60}")
        quick_duplicate_spotcheck(parts)

    combined = np.concatenate(list(parts.values()), axis=0)
    print(f"\nTotale reference 10MV: {len(combined):,} particelle da {len(parts)} part")

    with h5py.File(args.out, "w") as f:
        f.create_dataset("phase_space", data=combined, compression="gzip")
        f.attrs["n_total"] = len(combined)
        f.attrs["source_parts"] = list(parts.keys())
        f.attrs["note"] = ("Reference 10MV completamente held-out per l'esperimento "
                            "a sola energia (training su 6MV+25MV). Fascio aperto, "
                            "nessun jaw. Non bilanciato - statistica piena.")

    print(f"\nSalvato: {args.out}")


if __name__ == "__main__":
    main()
