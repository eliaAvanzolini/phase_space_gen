"""
prepare_conditional_data.py
============================
Converte i file IAEA (6MV + 10MV) in un unico dataset HDF5 condizionale
per l'addestramento dei modelli generativi (CFM, NSF, GAN).

Step:
    1. Legge i file IAEA 6MV (part1 + part2) → numpy
    2. Legge i file IAEA 10MV (part1 + part2) → numpy
    3. Bilancia i dataset (subsampling al più piccolo)
    4. Etichetta ciascuno con c = [E_nom, jaw_x, jaw_y]
    5. Shuffle e salva in data/elekta_multi_energy.h5

Per Strada B (dopo simulazione GATE con jaws):
    Lo script supporta anche l'aggiunta di file GATE ROOT
    con campi collimati (5×5, 10×10, 20×20) a qualsiasi energia.

Uso:
    # Step base: solo fascio aperto 6MV + 10MV (campo 10×10, jaw=5cm)
    python prepare_conditional_data.py

    # Con max_n per test rapido
    python prepare_conditional_data.py --max_n 100000

    # Dopo GATE: aggiungi anche i campi collimati
    python prepare_conditional_data.py \\
        --gate_phsp outputs/gate_jaw/6mv_5x5.root \\
        --gate_labels 6.0 2.5 2.5 \\
        --gate_phsp outputs/gate_jaw/6mv_10x10.root \\
        --gate_labels 6.0 5.0 5.0

    # Verifica
    python prepare_conditional_data.py --check data/elekta_multi_energy.h5
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.read_iaea_phsp import read_iaea_multi, read_iaea_phsp
from data.synthetic_linac import save_phase_space_hdf5, load_phase_space_hdf5


# ─── Configurazione dei file IAEA disponibili ────────────────────────────────

IAEA_SOURCES = {
    "6MV": {
        "E_nom": 6.0,
        "jaw_x": 5.0,  # campo aperto 10×10 → semi-apertura 5 cm
        "jaw_y": 5.0,
        "files": [
            "data/ELEKTA_PRECISE_6mv_part1_big.IAEAphsp",
            "data/ELEKTA_PRECISE_6mv_part2_big.IAEAphsp",
        ],
    },
    "10MV": {
        "E_nom": 10.0,
        "jaw_x": 5.0,
        "jaw_y": 5.0,
        "files": [
            "data/ELEKTA_PRECISE_10mv_part1.IAEAphsp",
            "data/ELEKTA_PRECISE_10mv_part2.IAEAphsp",
        ],
    },
}


def check_dataset(path: str) -> None:
    """Verifica la struttura e le statistiche del dataset HDF5."""
    import h5py

    print(f"\n{'='*60}")
    print(f"  Check dataset: {path}")
    print(f"{'='*60}")

    with h5py.File(path, "r") as f:
        print(f"\n  Keys nel file: {list(f.keys())}")

        ps = f["phase_space"][:]
        print(f"\n  phase_space shape: {ps.shape}  dtype: {ps.dtype}")

        cols = ["x [cm]", "y [cm]", "z [cm]", "dx", "dy", "dz", "E [MeV]"]
        print(f"\n  {'Canale':>10}  {'mu':>10}  {'σ':>10}  {'min':>10}  {'max':>10}")
        print(f"  {'-'*55}")
        for i, name in enumerate(cols):
            col = ps[:, i]
            print(f"  {name:>10}  {col.mean():>10.4f}  {col.std():>10.4f}  "
                  f"{col.min():>10.4f}  {col.max():>10.4f}")

        if "conditions" in f:
            cond = f["conditions"][:]
            print(f"\n  conditions shape: {cond.shape}  dtype: {cond.dtype}")
            print(f"  Colonne: [E_nom, jaw_x, jaw_y]")

            # Conta le configurazioni uniche
            unique_conds = np.unique(cond, axis=0)
            print(f"\n  Configurazioni uniche: {len(unique_conds)}")
            for uc in unique_conds:
                mask = np.all(cond == uc, axis=1)
                n = mask.sum()
                e_range = ps[mask, 6]
                print(f"    c=[{uc[0]:.1f}, {uc[1]:.1f}, {uc[2]:.1f}]: "
                      f"{n:>12,} campioni  "
                      f"E=[{e_range.min():.3f}, {e_range.max():.3f}] MeV")
        else:
            print("\n  [WARNING] Nessun vettore 'conditions' nel file.")

        if "metadata" in f:
            print(f"\n  Metadata:")
            for k, v in f["metadata"].attrs.items():
                print(f"    {k}: {v}")

    print(f"\n  ✓ Check completato")


def read_gate_root(root_path: str) -> np.ndarray:
    """
    Legge il file ROOT generato dal PhaseSpaceActor di OpenGATE 10.
    Converte le coordinate spaziali da mm (Geant4) a cm (allineamento IAEA).
    """
    import uproot
    print(f"  Estrazione ROOT da: {root_path}")
    with uproot.open(root_path) as f:
        # Recupera l'albero (OpenGATE nomina il tree come l'actor assegnato)
        tree = f["phsp_actor"]
        
        # Estrae i vettori come array NumPy
        arrays = tree.arrays([
            "PrePosition_X", "PrePosition_Y", "PrePosition_Z",
            "PreDirection_X", "PreDirection_Y", "PreDirection_Z",
            "KineticEnergy"
        ], library="np")
        
        # FISICA MEDICA CATCH: Geant4 esprime lo spazio in mm -> dividiamo per 10 per ottenere cm
        x_cm = arrays["PrePosition_X"] / 10.0
        y_cm = arrays["PrePosition_Y"] / 10.0
        z_cm = arrays["PrePosition_Z"] / 10.0
        
        dx = arrays["PreDirection_X"]
        dy = arrays["PreDirection_Y"]
        dz = arrays["PreDirection_Z"]
        E_mev = arrays["KineticEnergy"]
        
        return np.column_stack([x_cm, y_cm, z_cm, dx, dy, dz, E_mev]).astype(np.float32)


def main():
    p = argparse.ArgumentParser(
        description="Prepara dataset condizionale multi-energia da file IAEA e GATE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output", type=str, default="data/elekta_multi_energy.h5")
    p.add_argument("--max_n", type=int, default=None)
    p.add_argument("--no_balance", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check", type=str, default=None)
    # Parametri per i file GATE aggiuntivi (Strada B)
    p.add_argument("--gate_phsp", type=str, nargs="+", default=None)
    p.add_argument("--gate_labels", type=float, nargs=3, action="append", default=None)
    args = p.parse_args()

    if args.check:
        check_dataset(args.check)
        return

    # 1. Costruisci la base iniziale caricando i file IAEA standard
    output_path = args.output
    rng = np.random.default_rng(args.seed)
    
    all_ps = []
    all_cond = []
    counts = {}

    print(f"\n{'='*60}")
    print(f"  COMPILAZIONE SUPER-DATASET CONDIZIONALE (STRADA B)")
    print(f"{'='*60}")

    # Estrazione sorgenti IAEA fisse
    for name, src in IAEA_SOURCES.items():
        existing_files = [f for f in src["files"] if Path(f).exists()]
        if not existing_files:
            continue
        
        if len(existing_files) > 1:
            ps = read_iaea_multi(existing_files, max_n_per_file=args.max_n)
        else:
            ps = read_iaea_phsp(existing_files[0], max_n=args.max_n)
            
        c = np.tile([src["E_nom"], src["jaw_x"], src["jaw_y"]], (len(ps), 1)).astype(np.float32)
        all_ps.append(ps)
        all_cond.append(c)
        counts[name] = len(ps)

    # 2. SEZIONE CORRETTA: Estrazione e integrazione dei file collimati da GATE
    if args.gate_phsp and args.gate_labels:
        if len(args.gate_phsp) != len(args.gate_labels):
            print("[ERROR] Il numero di file --gate_phsp deve coincidere con il numero di --gate_labels.")
            sys.exit(1)
            
        for root_file, labels in zip(args.gate_phsp, args.gate_labels):
            if not Path(root_file).exists():
                print(f"  [WARNING] File GATE non trovato: {root_file}. Skipping.")
                continue
                
            ps_gate = read_gate_root(root_file)
            if args.max_n:
                ps_gate = ps_gate[:args.max_n]
                
            c_gate = np.tile(labels, (len(ps_gate), 1)).astype(np.float32)
            
            all_ps.append(ps_gate)
            all_cond.append(c_gate)
            label_str = f"GATE_{labels[0]}mv_{2*labels[1]:.0f}x{2*labels[2]:.0f}"
            counts[label_str] = len(ps_gate)

    if not all_ps:
        print("[ERROR] Nessun dato caricato (né IAEA né GATE). Verifica i percorsi.")
        sys.exit(1)

    # 3. Bilanciamento delle classi (Subsampling protetto al minimo per evitare bias)
    if not args.no_balance and len(counts) > 1:
        min_count = min(counts.values())
        print(f"\n  Bilanciamento attivo: subsampling a {min_count:,} eventi per configurazione")
        balanced_ps, balanced_cond = [], []
        for dataset_ps, dataset_c in zip(all_ps, all_cond):
            idx = rng.choice(len(dataset_ps), size=min_count, replace=False)
            balanced_ps.append(dataset_ps[idx])
            balanced_cond.append(dataset_c[idx])
        all_ps, all_cond = balanced_ps, balanced_cond

    # Concatena e applica uno shuffle globale per mescolare le configurazioni nei batch
    ps_combined = np.concatenate(all_ps, axis=0)
    cond_combined = np.concatenate(all_cond, axis=0)
    
    perm = rng.permutation(len(ps_combined))
    ps_combined = ps_combined[perm]
    cond_combined = cond_combined[perm]

    # Salvataggio HDF5 centralizzato con metadati trasparenti
    metadata = {"balanced": str(not args.no_balance), "n_total": str(len(ps_combined))}
    for name, count in counts.items():
        metadata[f"n_{name}"] = str(count)
        
    save_phase_space_hdf5(ps_combined, cond_combined, output_path, metadata=metadata)
    check_dataset(output_path)

if __name__ == "__main__":
    main()