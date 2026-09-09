import uproot
import numpy as np
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--field", required=True)
parser.add_argument("--dir", required=True)
args = parser.parse_args()

files = glob.glob(f"{args.dir}/*_phsp_part*.root")
if not files:
    print(f"Nessun file trovato in {args.dir}")
    exit(1)
test_file = files[0]

with uproot.open(test_file) as f:
    tree = f[f.keys()[0]]
    x = np.abs(tree["PrePosition_X"].array(library="np")) / 10.0  # mm -> cm
    y = np.abs(tree["PrePosition_Y"].array(library="np")) / 10.0

    # Campo nominale isocentrico (Z=100cm) per questa classe
    jaw_size_iso = float(args.field.split("_")[1].split("x")[0]) / 2.0

    # FIX: il piano phsp e' a Z=50cm, non all'isocentro (Z=100cm). Il fascio
    # diverge da un fuoco a Z=0, quindi la stessa proiezione per triangoli
    # simili usata per i jaw (d(Z) = d_iso * Z/100) va applicata anche qui:
    # a Z=50cm la semiapertura REALE del fascio collimato e' meta' di quella
    # isocentrica. Senza questo fattore, la soglia e' troppo larga e il leak
    # misurato risulta sistematicamente sottostimato.
    PHSP_Z = 50.0
    ISOCENTER_Z = 100.0
    jaw_size_at_phsp = jaw_size_iso * (PHSP_Z / ISOCENTER_Z)

    outside = (x > jaw_size_at_phsp) | (y > jaw_size_at_phsp)
    leak = (np.sum(outside) / len(x)) * 100

    print(f"File analizzato: {test_file}")
    print(f"Totale fotoni: {len(x):,}")
    print(f"Semiapertura nominale isocentrica: {jaw_size_iso:.2f} cm")
    print(f"Semiapertura attesa a Z={PHSP_Z:.0f}cm (proiettata): {jaw_size_at_phsp:.2f} cm")
    print(f"Leakage geometrico (> {jaw_size_at_phsp:.2f}cm a Z={PHSP_Z:.0f}cm): {leak:.2f}%")
