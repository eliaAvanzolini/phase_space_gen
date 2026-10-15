
#!/usr/bin/env python3
"""
Diagnostica il flag 'first' nei phase space ASCII TOPAS.

Per ogni file .phsp calcola:
- numero totale di particelle registrate
- numero e frazione di righe con first = 1
- conteggio delle storie che hanno raggiunto il piano, se presente nell'header
- differenza fra conteggio first=1 e storie raggiunte
- distribuzione del flag per specie PDG

ATTENZIONE:
Il flag first identifica la prima particella registrata di una storia,
non identifica direttamente una particella primaria o secondaria.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# Prime 10 colonne standard del phase space ASCII TOPAS.
COLS = [
    "x", "y", "z", "dx", "dy", "E",
    "weight", "pdg", "dz_neg", "first",
]

PDG_NAMES = {
    2212: "proton",
    11: "electron",
    -11: "positron",
    22: "photon",
    2112: "neutron",
    1000010020: "deuteron",
    1000020040: "alpha",
}


def read_header_info(phsp_path):
    """Legge i conteggi principali dal file .header associato."""
    header_path = phsp_path.with_suffix(".header")

    info = {
        "header_n_histories": np.nan,
        "header_n_histories_reached": np.nan,
        "header_n_scored_particles": np.nan,
        "header_exists": header_path.exists(),
    }

    if not header_path.exists():
        return info

    text = header_path.read_text(errors="replace")

    patterns = {
        "header_n_histories":
            r"Number of Original Histories:\s*(\d+)",
        "header_n_histories_reached":
            r"Number of Original Histories that Reached Phase Space:\s*(\d+)",
        "header_n_scored_particles":
            r"Number of Scored Particles:\s*(\d+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            info[key] = int(match.group(1))

    return info


def analyse_file(phsp_path, chunksize=250_000):
    """Analizza un .phsp in streaming, senza caricare tutto in memoria."""
    phsp_path = Path(phsp_path)

    if not phsp_path.exists():
        raise FileNotFoundError(phsp_path)

    header = read_header_info(phsp_path)

    n_rows = 0
    n_first_1 = 0
    n_first_0 = 0
    n_invalid_first = 0

    # Conteggi indipendenti per ciascun codice PDG.
    species_rows = defaultdict(int)
    species_first_1 = defaultdict(int)

    # Si leggono soltanto le colonne PDG e first: basta per questa diagnosi.
    reader = pd.read_csv(
        phsp_path,
        sep=r"\s+",
        header=None,
        names=COLS,
        usecols=["pdg", "first"],
        dtype={"pdg": "int64", "first": "float64"},
        chunksize=chunksize,
        engine="c",
    )

    for chunk in reader:
        pdg = chunk["pdg"].to_numpy()
        first = chunk["first"].to_numpy(dtype=np.float64)

        n_rows += len(chunk)

        is_first = first == 1
        is_not_first = first == 0
        is_valid = is_first | is_not_first

        n_first_1 += int(is_first.sum())
        n_first_0 += int(is_not_first.sum())
        n_invalid_first += int((~is_valid).sum())

        # Statistiche separate per specie.
        for code in np.unique(pdg):
            mask = pdg == code
            species_rows[int(code)] += int(mask.sum())
            species_first_1[int(code)] += int(
                (mask & is_first).sum()
            )

    frac_first = n_first_1 / n_rows if n_rows else np.nan

    n_histories_reached = header["header_n_histories_reached"]
    n_scored_header = header["header_n_scored_particles"]

    if np.isfinite(n_histories_reached):
        delta_first_vs_histories = (
            n_first_1 - int(n_histories_reached)
        )
    else:
        delta_first_vs_histories = np.nan

    if np.isfinite(n_scored_header):
        delta_rows_vs_header = n_rows - int(n_scored_header)
    else:
        delta_rows_vs_header = np.nan

    summary = {
        "file": phsp_path.name,
        "path": str(phsp_path),
        "header_exists": header["header_exists"],
        "n_rows_phsp": n_rows,
        "n_first_1_all_species": n_first_1,
        "frac_first_all_species": frac_first,
        "n_first_0_all_species": n_first_0,
        "n_invalid_first": n_invalid_first,
        "header_n_histories": header["header_n_histories"],
        "header_n_histories_reached": n_histories_reached,
        "header_n_scored_particles": n_scored_header,
        "first1_minus_histories_reached": delta_first_vs_histories,
        "rows_minus_header_scored": delta_rows_vs_header,
        "n_protons": species_rows.get(2212, 0),
        "n_first1_protons": species_first_1.get(2212, 0),
        "frac_first_protons": (
            species_first_1[2212] / species_rows[2212]
            if species_rows.get(2212, 0) else np.nan
        ),
    }

    species_results = []

    for code in sorted(species_rows):
        count = species_rows[code]
        count_first = species_first_1[code]

        species_results.append({
            "file": phsp_path.name,
            "pdg": code,
            "species": PDG_NAMES.get(code, "other/ion"),
            "n_rows": count,
            "n_first_1": count_first,
            "frac_first": count_first / count if count else np.nan,
        })

    return summary, species_results


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostica il flag first dei phase space TOPAS."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="File .phsp da analizzare (puoi passarne diversi).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Analizza ricorsivamente tutti i .phsp sotto questa directory.",
    )
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument(
        "--out-summary",
        default="data/metadata/first_flag_file_summary.csv",
    )
    parser.add_argument(
        "--out-species",
        default="data/metadata/first_flag_species_summary.csv",
    )
    args = parser.parse_args()

    if args.root is not None and args.files:
        parser.error("Usa i file espliciti oppure --root, non entrambi.")

    if args.root is not None:
        files = sorted(args.root.rglob("*.phsp"))
    else:
        files = [Path(p) for p in args.files]

    if not files:
        parser.error("Specifica almeno un .phsp oppure usa --root.")

    summaries = []
    species_summaries = []

    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Analizzo {path} ...", flush=True)

        try:
            summary, species_rows = analyse_file(
                path, chunksize=args.chunksize
            )
            summaries.append(summary)
            species_summaries.extend(species_rows)

            print(
                f"  righe={summary['n_rows_phsp']:,} | "
                f"first=1: {summary['n_first_1_all_species']:,} "
                f"({summary['frac_first_all_species']:.3%}) | "
                f"protoni: {summary['n_protons']:,} | "
                f"first=1 nei protoni: {summary['frac_first_protons']:.3%}"
            )

            if np.isfinite(summary["first1_minus_histories_reached"]):
                print(
                    "  first=1 meno storie raggiunte: "
                    f"{summary['first1_minus_histories_reached']:,.0f}"
                )

        except Exception as exc:
            print(f"  ERRORE: {exc}")

    if not summaries:
        raise SystemExit("Nessun file analizzato con successo.")

    summary_df = pd.DataFrame(summaries)
    species_df = pd.DataFrame(species_summaries)

    for output in (args.out_summary, args.out_species):
        Path(output).parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(args.out_summary, index=False)
    species_df.to_csv(args.out_species, index=False)

    print("\nFile riepilogo:", args.out_summary)
    print("File per specie:", args.out_species)
    print("\nDiagnosi completata.")


if __name__ == "__main__":
    main()
