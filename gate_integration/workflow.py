"""
gate_integration/workflow.py
==============================
Pipeline end-to-end completa per la tesi.

Esegue in sequenza le 4 fasi della roadmap:

    Fase 1: Baseline GAN (parametri paper Sarrut 2019)
    Fase 2: NSF 6D (Neural Spline Flow)
    Fase 3: CFM condizionato (Conditional Flow Matching)
    Fase 4: Validazione downstream con GATE 10

Uso tipico sulla workstation:

    # Fase 1+2+3 con dati sintetici (no GATE, per sviluppo)
    python gate_integration/workflow.py --synthetic --phases 1 2 3

    # Fase 1+2+3 con dati GATE reali
    python gate_integration/workflow.py \\
        --hdf5_data data/linac_6MV_train.h5 \\
        --phases 1 2 3

    # Fase 4: validazione downstream (richiede GATE installato)
    python gate_integration/workflow.py \\
        --hdf5_data data/linac_6MV_train.h5 \\
        --phases 4 \\
        --phsp_reference data/linac_6MV_eval.root

    # Tutto insieme
    python gate_integration/workflow.py \\
        --hdf5_data data/linac_6MV_train.h5 \\
        --phases 1 2 3 4 \\
        --phsp_reference data/linac_6MV_eval.root
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_phase1_gan(hdf5_data, synthetic, output_root, n_samples, n_epochs):
    """Fase 1: Addestra baseline GAN (parametri esatti del paper Sarrut 2019)."""
    import subprocess

    print(f"\n{'='*55}")
    print(f"  FASE 1: Baseline GAN (Sarrut 2019)")
    print(f"  Parametri: H=400, z_dim=6, lr=1e-5, RMSProp")
    print(f"  Epoche: {n_epochs} | Paper: 80,000")
    print(f"{'='*55}")

    cmd = [sys.executable, "baseline_gaga.py",
           "--output_dir", str(output_root / "phase1_gan"),
           "--n_epochs", str(n_epochs),
           "--batch_size", "10000",  # paper: 10,000
           "--h_dim", "400",         # paper: H=400
           "--z_dim", "6",           # paper: z_dim=6
           "--lr", "1e-5",           # paper: lr=1e-5
           "--n_critic", "4",        # paper: 4 critic / 1 generator
           "--log_every", str(max(100, n_epochs // 100)),
           "--save_every", str(max(500, n_epochs // 20)),
    ]

    if synthetic:
        cmd.append("--synthetic")
    else:
        cmd.extend(["--hdf5_train", str(hdf5_data)])

    if n_samples:
        cmd.extend(["--n_train", str(n_samples)])

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    return result.returncode == 0


def run_phase2_nsf(hdf5_data, synthetic, output_root, n_samples, n_epochs):
    """Fase 2: Neural Spline Flow (confronto diretto con GAN)."""
    import subprocess

    print(f"\n{'='*55}")
    print(f"  FASE 2: Neural Spline Flow")
    print(f"  Architettura: K=6 coupling, 8 bin, hidden=128")
    print(f"{'='*55}")

    cmd = [sys.executable, "train.py",
           "--model", "nsf",
           "--n_transforms", "6",
           "--n_bins", "8",
           "--hidden_dim", "128",
           "--lr", "1e-3",
           "--batch_size", "2048",
           "--epochs", str(n_epochs),
           "--output_dir", str(output_root / "phase2_nsf"),
    ]

    if synthetic:
        cmd.append("--synthetic")
        cmd.extend(["--n_samples", str(n_samples or 1_000_000)])
    else:
        cmd.extend(["--data_path", str(hdf5_data)])

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    return result.returncode == 0


def run_phase3_cfm(hdf5_data, synthetic, output_root, n_samples, n_epochs):
    """Fase 3: Conditional Flow Matching (modello condizionato multi-sorgente)."""
    import subprocess

    print(f"\n{'='*55}")
    print(f"  FASE 3: Conditional Flow Matching")
    print(f"  Condizionato su: E_nom, jaw_x, jaw_y")
    print(f"{'='*55}")

    cmd = [sys.executable, "train.py",
           "--model", "cfm",
           "--conditional",
           "--n_layers", "4",
           "--hidden_dim", "256",
           "--lr", "1e-3",
           "--batch_size", "4096",
           "--epochs", str(n_epochs),
           "--output_dir", str(output_root / "phase3_cfm"),
    ]

    if synthetic:
        cmd.append("--synthetic")
        cmd.extend(["--n_samples", str(n_samples or 1_000_000)])
    else:
        cmd.extend(["--data_path", str(hdf5_data)])

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    return result.returncode == 0


def run_phase4_gate(output_root, phsp_reference, n_particles):
    """
    Fase 4: Validazione downstream con GATE 10.

    Per ogni modello addestrato nelle fasi 1-3:
    1. Salva il modello in formato .pth per GATE
    2. Esegue simulazione dose con GANSource
    3. Calcola gamma-index vs gold standard
    """
    import subprocess

    print(f"\n{'='*55}")
    print(f"  FASE 4: Validazione Downstream (GATE 10)")
    print(f"{'='*55}")

    results = {}

    # Trova tutti i modelli addestrati
    models_to_validate = []
    for phase_dir, model_type in [
        ("phase1_gan", "gan"),
        ("phase2_nsf", "nsf"),
        ("phase3_cfm", "cfm"),
    ]:
        run_dirs = sorted((output_root / phase_dir).glob("*/best_model.pt"))
        if run_dirs:
            latest = run_dirs[-1].parent  # ultima run
            models_to_validate.append((model_type, latest))

    if not models_to_validate:
        print("  Nessun modello trovato. Eseguire prima le fasi 1-3.")
        return {}

    # Genera dose reference dal phase space GATE (se disponibile)
    ref_dose_path = None
    if phsp_reference and Path(phsp_reference).exists():
        ref_dir = output_root / "phase4_reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            sys.executable, "gate_integration/gate_simulations.py",
            "dose_reference",
            "--phsp_file", str(phsp_reference),
            "--n_particles", str(int(n_particles)),
            "--output_dir", str(ref_dir),
        ], cwd=str(Path(__file__).parent.parent))
        ref_dose_path = ref_dir / "dose_reference.mhd"

    # Per ogni modello: salva .pth → simula dose → calcola gamma
    for model_type, run_dir in models_to_validate:
        print(f"\n  Validazione: {model_type.upper()} ({run_dir})")

        stats_path = run_dir / "normalization_stats.json"
        if not stats_path.exists():
            print(f"    stats non trovate: {stats_path} — skip")
            continue

        # Salva in formato GATE
        pth_path = run_dir / f"{model_type}_gate.pth"
        subprocess.run([
            sys.executable, "gate_integration/save_for_gate.py",
            "--checkpoint",  str(run_dir / "best_model.pt"),
            "--model",       model_type,
            "--stats_path",  str(stats_path),
            "--out",         str(pth_path),
        ], cwd=str(Path(__file__).parent.parent))

        if not pth_path.exists():
            print(f"    .pth non generato — skip")
            continue

        # Simula dose con GANSource
        dose_dir = output_root / f"phase4_{model_type}"
        dose_dir.mkdir(parents=True, exist_ok=True)

        subprocess.run([
            sys.executable, "gate_integration/gate_simulations.py",
            "dose_model",
            "--pth_filename", str(pth_path),
            "--n_particles",  str(int(n_particles)),
            "--output_dir",   str(dose_dir),
        ], cwd=str(Path(__file__).parent.parent))

        # Gamma-index vs reference
        model_dose = dose_dir / f"dose_{model_type}_gate.mhd"
        if ref_dose_path and ref_dose_path.exists() and model_dose.exists():
            gi_out = dose_dir / "gamma_index.json"
            subprocess.run([
                sys.executable, "gate_integration/gate_simulations.py",
                "gamma_index",
                "--reference", str(ref_dose_path),
                "--model",     str(model_dose),
                "--out_json",  str(gi_out),
            ], cwd=str(Path(__file__).parent.parent))

            if gi_out.exists():
                with open(gi_out) as f:
                    results[model_type] = json.load(f)

    # Stampa tabella comparativa
    if results:
        print(f"\n{'='*55}")
        print(f"  RISULTATI GAMMA-INDEX (2%/2mm)")
        print(f"{'='*55}")
        print(f"  {'Modello':<10} {'Pass Rate':>12} {'Mean Δ%':>10} {'Max |Δ|%':>12}")
        print(f"  {'-'*46}")
        for model, r in results.items():
            print(f"  {model.upper():<10} "
                  f"{r.get('pass_rate_pct', 0):>11.2f}% "
                  f"{r.get('mean_diff_pct', 0):>+10.3f} "
                  f"{r.get('max_abs_diff_pct', 0):>11.3f}%")

    return results


def compare_all_models(output_root):
    """Confronta le metriche di tutti i modelli addestrati e genera la tabella finale."""
    from evaluate import compare_models
    from data.synthetic_linac import generate_phase_space, denormalize_phase_space

    print(f"\n{'='*55}")
    print(f"  CONFRONTO FINALE: GAN vs NSF vs CFM")
    print(f"{'='*55}")

    # Genera un dataset di riferimento sintetico per il confronto
    real = generate_phase_space(100_000, seed=999)

    models_gen = {}
    for phase_dir, model_type in [("phase1_gan", "gan"), ("phase2_nsf", "nsf"), ("phase3_cfm", "cfm")]:
        samples_path = list((output_root / phase_dir).glob("*/generated_ps.h5"))
        if samples_path:
            import h5py
            with h5py.File(samples_path[-1]) as f:
                gen = f["phase_space"][:]
            models_gen[model_type.upper()] = gen

    if len(models_gen) >= 2:
        compare_models(real, models_gen, str(output_root / "comparison"))


def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline end-to-end: GAN → NSF → CFM → GATE validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dati
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic",   action="store_true",
                   help="Usa dati sintetici (no GATE richiesto)")
    g.add_argument("--hdf5_data",   type=str,
                   help="File HDF5 con dati GATE reali")

    p.add_argument("--phases",      type=int, nargs="+", default=[1, 2, 3],
                   choices=[1, 2, 3, 4],
                   help="Fasi da eseguire (1=GAN, 2=NSF, 3=CFM, 4=GATE dose)")

    # Parametri di training
    p.add_argument("--n_samples",   type=int, default=1_000_000,
                   help="Campioni di training (solo per dati sintetici)")
    p.add_argument("--epochs_gan",  type=int, default=80_000,
                   help="Epoche GAN (paper: 80000, quick test: 5000)")
    p.add_argument("--epochs_nsf",  type=int, default=200,
                   help="Epoche NSF")
    p.add_argument("--epochs_cfm",  type=int, default=300,
                   help="Epoche CFM")

    # Fase 4
    p.add_argument("--phsp_reference", type=str, default=None,
                   help="[Fase 4] File ROOT phase space di riferimento GATE")
    p.add_argument("--n_particles",    type=float, default=1e7,
                   help="[Fase 4] Particelle per le simulazioni dose GATE")

    p.add_argument("--output_dir",  type=str, default="outputs/workflow",
                   help="Directory base per tutti gli output")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    phase_results = {}

    if 1 in args.phases:
        ok = run_phase1_gan(
            args.hdf5_data, args.synthetic, output_root,
            args.n_samples, args.epochs_gan
        )
        phase_results["phase1_gan"] = "OK" if ok else "FAILED"

    if 2 in args.phases:
        ok = run_phase2_nsf(
            args.hdf5_data, args.synthetic, output_root,
            args.n_samples, args.epochs_nsf
        )
        phase_results["phase2_nsf"] = "OK" if ok else "FAILED"

    if 3 in args.phases:
        ok = run_phase3_cfm(
            args.hdf5_data, args.synthetic, output_root,
            args.n_samples, args.epochs_cfm
        )
        phase_results["phase3_cfm"] = "OK" if ok else "FAILED"

    if 4 in args.phases:
        gate_results = run_phase4_gate(
            output_root, args.phsp_reference, args.n_particles
        )
        phase_results["phase4_gate"] = gate_results

    # Confronto finale
    if set(args.phases) >= {1, 2, 3}:
        compare_all_models(output_root)

    elapsed = time.time() - t_total
    print(f"\n{'='*55}")
    print(f"  PIPELINE COMPLETATA in {elapsed/60:.1f} min")
    for phase, status in phase_results.items():
        print(f"  {phase}: {status}")
    print(f"  Output: {output_root}")
    print(f"{'='*55}")
