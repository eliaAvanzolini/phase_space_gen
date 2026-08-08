import uproot
import numpy as np

parts = [
    "data/ELEKTA_PRECISE_10mv_part1.root",
    "data/ELEKTA_PRECISE_10mv_part2.root",
    "data/ELEKTA_PRECISE_10mv_part3.root",
    "data/ELEKTA_PRECISE_10mv_part4.root",
]

print("=" * 60)
print(" CONTROLLO Z PER SINGOLA PARTE (10MV)")
print("=" * 60)

for p in parts:
    with uproot.open(p) as f:
        tree = f[f.keys()[0]]
        z = tree["PrePosition_Z"].array(library="np")
    print(f"{p}")
    print(f"  N = {len(z):,}")
    print(f"  Z mean = {z.mean():.6f}   Z std = {z.std():.6f}")
    print(f"  Z min  = {z.min():.6f}   Z max = {z.max():.6f}\n")
