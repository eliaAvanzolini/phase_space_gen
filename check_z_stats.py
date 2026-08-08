# check_z_stats.py
import h5py
import numpy as np
import time

filename = "data/energy_only_train_dataset.h5"

print("=" * 60)
print(" ISPEZIONE STATISTICHE Z PER ENERGIA")
print("=" * 60)

t0 = time.time()

with h5py.File(filename, "r") as f:
    print("Caricamento colonna condizioni (E_nom)...")
    cond_e = f["conditions"][:, 0]

    print("Caricamento colonna Z dal phase space...")
    z_all = f["phase_space"][:, 2]

print(f"Dati letti in {time.time() - t0:.2f} secondi.\n")

for e in [6.0, 25.0]:
    mask = np.abs(cond_e - e) < 0.1
    z_vals = z_all[mask]

    if len(z_vals) > 0:
        print(f"--- Energia {e} MeV ---")
        print(f"  Vettori trovati : {len(z_vals):,}")
        print(f"  Z Mean          : {z_vals.mean():.6f} cm")
        print(f"  Z Std           : {z_vals.std():.6f} cm")
        print(f"  Z Min           : {z_vals.min():.6f} cm")
        print(f"  Z Max           : {z_vals.max():.6f} cm\n")
    else:
        print(f"--- Energia {e} MeV: Nessun dato trovato ---\n")

print("Completato!")
