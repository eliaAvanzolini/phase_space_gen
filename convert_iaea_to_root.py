"""
data/convert_iaea_to_root.py
============================
Converte un file binario IAEA (.IAEAphsp) in un file .root compatibile con OpenGATE 10.
Converte le distanze da cm (IAEA) a mm (Geant4/GATE) mappando sia branch locali che globali.
"""
import argparse
import numpy as np
import uproot
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.read_iaea_phsp import read_iaea_phsp, _find_header, parse_iaea_header

def main():
    p = argparse.ArgumentParser(description="Converte IAEA phsp in GATE ROOT")
    p.add_argument("--input", required=True, help="Path del file .IAEAphsp")
    p.add_argument("--output", required=True, help="Path del file .root di output")
    p.add_argument("--max_n", type=int, default=None)
    args = p.parse_args()

    # 1. Leggi i dati binari IAEA via NumPy
    print(f"  [1/3] Decodifica file IAEA: {args.input}")
    ps = read_iaea_phsp(args.input, max_n=args.max_n)

    # 2. Scrittura del file ROOT con mappatura doppia (Global + Local) per OpenGATE 10
    print(f"  [2/3] Scrittura albero ROOT (conversione cm -> mm): {args.output}")
    x_mm = ps[:, 0] * 10.0
    y_mm = ps[:, 1] * 10.0
    z_mm = ps[:, 2] * 10.0
    
    with uproot.recreate(args.output) as f:
        f["phsp_actor"] = {
            # Coordinate Globali
            "PrePosition_X": x_mm,
            "PrePosition_Y": y_mm,
            "PrePosition_Z": z_mm,
            # Coordinate Locali (FIX per OpenGATE 10 PhaseSpaceSource)
            "PrePositionLocal_X": x_mm,
            "PrePositionLocal_Y": y_mm,
            "PrePositionLocal_Z": z_mm,
            # Direzioni Globali e Locali
            "PreDirection_X": ps[:, 3],
            "PreDirection_Y": ps[:, 4],
            "PreDirection_Z": ps[:, 5],
            "PreDirectionLocal_X": ps[:, 3],
            "PreDirectionLocal_Y": ps[:, 4],
            "PreDirectionLocal_Z": ps[:, 5],
            # Energia e Nome
            "KineticEnergy": ps[:, 6],
            "ParticleName": ["gamma"] * len(ps)
        }
    print(f"  [3/3] ✓ Conversione completata con successo!")

if __name__ == "__main__":
    main()
