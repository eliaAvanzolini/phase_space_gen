"""
build_depth_dataset.py
======================
Costruisce l'HDF5 per un modello condizionato sulla profondità d, a partire dalla cartella
dei SurfaceFile_<depth>_<peak|valley>_1_2_5.phsp.  Layout come data/dataset.py:
    phase_space (N,7) = (x,y,z,dx,dy,dz,E)     conditions (N,1) = [depth_cm]

Per default usa SOLO i file "peak" (i "valley" vengono ignorati, come da indicazione del prof)
e solo i protoni. La profondità è presa dal nome file e verificata contro la z del piano
(z = -(158 + depth)); se i due non coincidono avvisa.

Esempi
------
    # inventario (nessun file scritto): N protoni, E mean/std, x/y std per profondità
    python build_depth_dataset.py --dir /percorso/phsp --inventory

    # train su tutte le profondità tranne 2.5 e 3.5, che finiscono nel file di hold-out per il test di interpolazione
    python build_depth_dataset.py --dir /percorso/phsp --holdout 2.5 3.5 --out data/proton_depth_train.h5

    # sottocampiona ogni profondità a N protoni (le statistiche per file sono diverse)
    python build_depth_dataset.py --dir /percorso/phsp --balance --out data/proton_depth.h5
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from read_topas_phsp import (PROTON, depth_from_z, load_filtered, parse_name, read_phsp)


def find_files(folder, region):
    out = []
    for f in sorted(Path(folder).glob("*.phsp")):
        info = parse_name(f.name)
        if info is None:
            print(f"[skip] nome non riconosciuto: {f.name}")
        elif region != "all" and info["region"] != region:
            continue
        else:
            out.append((info["depth"], info["region"], f))
    return sorted(out, key=lambda t: (t[0], t[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="cartella con i .phsp")
    ap.add_argument("--region", default="peak", choices=["peak", "valley", "all"])
    ap.add_argument("--out", default=None, help="HDF5 di training (obbligatorio se non --inventory)")
    ap.add_argument("--inventory", action="store_true", help="solo tabella riassuntiva per profondità")
    ap.add_argument("--holdout", nargs="*", type=float, default=[],
                    help="profondità [cm] da tenere fuori dal training -> <out>_holdout.h5")
    ap.add_argument("--input", default=None, help="Input_70MeV.phsp, salvato come 'input_ps' nel file")
    ap.add_argument("--all_species", action="store_true", help="non filtrare sui soli protoni")
    ap.add_argument("--balance", action="store_true", help="sottocampiona ogni profondità al minimo comune")
    ap.add_argument("--max_per_depth", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if not a.inventory and not a.out:
        ap.error("serve --out (oppure --inventory)")

    files = find_files(a.dir, a.region)
    if not files:
        raise SystemExit("Nessun file trovato.")
    rng = np.random.default_rng(a.seed)

    rows, data = [], {}
    for depth, region, f in files:
        df = read_phsp(f)
        z_unique = np.unique(df["z"])
        if len(z_unique) != 1 or abs(depth_from_z(z_unique[0]) - depth) > 1e-3:
            print(f"[ATTENZIONE] {f.name}: nome -> depth={depth}, ma z={z_unique[:3]} -> depth_z={depth_from_z(z_unique[0]):.3f}")
        n_all = len(df)
        if not a.all_species:
            df = df[df["pdg"] == PROTON]
        E = df["E"].to_numpy()
        rows.append(dict(depth=depth, region=region, n_total=n_all, n_kept=len(df),
                         E_mean=float(E.mean()), E_std=float(E.std()),
                         x_std=float(df["x"].std()), y_std=float(df["y"].std()), file=f.name))
        if not a.inventory:
            from read_topas_phsp import to_7d
            data[(depth, region)] = to_7d(df)

    print(f"\n{'depth':>6} {'reg':>6} {'N_tot':>9} {'N_kept':>9} {'E_mean':>8} {'E_std':>7} {'x_std':>7} {'y_std':>7}")
    for r in rows:
        print(f"{r['depth']:6.2f} {r['region']:>6} {r['n_total']:9d} {r['n_kept']:9d} "
              f"{r['E_mean']:8.2f} {r['E_std']:7.2f} {r['x_std']:7.3f} {r['y_std']:7.3f}")
    if a.inventory:
        return

    train, hold = [], []
    for (d, reg), ps in data.items():
        if a.max_per_depth and len(ps) > a.max_per_depth:
            ps = ps[rng.choice(len(ps), a.max_per_depth, replace=False)]
        (hold if any(abs(d - h) < 1e-6 for h in a.holdout) else train).append((d, ps))
    if a.balance and train:
        n_min = min(len(p) for _, p in train)
        train = [(d, p[rng.choice(len(p), n_min, replace=False)]) for d, p in train]
        print(f"\nbilanciato a {n_min} protoni per profondità (train)")

    def write(path, chunks, with_input=False):
        ps = np.concatenate([p for _, p in chunks]).astype(np.float32)
        cond = np.concatenate([np.full((len(p), 1), d, np.float32) for d, p in chunks])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as h:
            h.create_dataset("phase_space", data=ps, compression="gzip")
            h.create_dataset("conditions", data=cond, compression="gzip")
            h.attrs["depths"] = json.dumps(sorted({d for d, _ in chunks}))
            if with_input and a.input:
                inp, _ = load_filtered(a.input, protons_only=not a.all_species)
                h.create_dataset("input_ps", data=inp, compression="gzip")
        print(f"scritto {path}: phase_space {ps.shape}, depths={sorted({d for d, _ in chunks})}")

    write(a.out, train, with_input=True)
    if hold:
        write(str(Path(a.out).with_name(Path(a.out).stem + "_holdout.h5")), hold)


if __name__ == "__main__":
    main()
