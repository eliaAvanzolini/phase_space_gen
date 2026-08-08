import sys
import argparse
from pathlib import Path

# Aggiunge sia la root che la cartella data/ al path di Python
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "data"))

import uproot
from read_iaea_phsp import read_iaea_phsp


def convert(energy: str, part: str):
    input_phsp = f"data/ELEKTA_PRECISE_{energy}_{part}.IAEAphsp"
    output_root = f"data/ELEKTA_PRECISE_{energy}_{part}.root"

    if not Path(input_phsp).exists():
        print(f"❌ File non trovato: {input_phsp} (trasferimento completato?)")
        sys.exit(1)

    print(f"🚀 [JOB] Leggo il file IAEA {energy.upper()} {part}: {input_phsp}...")
    ps_data = read_iaea_phsp(input_phsp, remove_511=True)
    print(f"\n✅ Lettura completata! Campioni estratti e filtrati: {len(ps_data):,}")

    print(f"💾 Scrittura nel file ROOT: {output_root}...")
    branch_dict = {
        "PrePosition_X": ps_data[:, 0],
        "PrePosition_Y": ps_data[:, 1],
        "PrePosition_Z": ps_data[:, 2],
        "PreDirection_X": ps_data[:, 3],
        "PreDirection_Y": ps_data[:, 4],
        "PreDirection_Z": ps_data[:, 5],
        "KineticEnergy": ps_data[:, 6],
    }
    with uproot.recreate(output_root) as f:
        f["PhaseSpace"] = branch_dict

    print(f"🎉 Conversione {energy.upper()} {part} terminata! Creato {output_root} "
          f"con {len(ps_data):,} particelle.")
    return len(ps_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy", choices=["6mv", "10mv"], required=True)
    parser.add_argument("--part", choices=["part1", "part2", "part3", "part4"], required=True)
    args = parser.parse_args()
    convert(args.energy, args.part)
