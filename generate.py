"""
generate.py
===========
Script di inferenza: carica un modello addestrato e genera un file di
phase space pronto per essere usato come sorgente in GATE.

Uso:
    # Genera 1M campioni con il modello CFM (configurazione 6MV 10x10)
    python generate.py --checkpoint outputs/cfm_run/best_model.pt \\
                       --model cfm \\
                       --n_samples 1000000 \\
                       --E_nom 6.0 --jaw_x 5.0 --jaw_y 5.0 \\
                       --out generated_ps.h5

    # Genera con NSF non condizionato
    python generate.py --checkpoint outputs/nsf_run/best_model.pt \\
                       --model nsf --n_samples 500000 --out ps_nsf.h5

    # Genera con velocità massima (Euler, 10 step) per produzione
    python generate.py --checkpoint outputs/cfm_run/best_model.pt \\
                       --model cfm --fast --n_samples 5000000 --out ps_prod.h5

Output:
    File HDF5 compatibile con GATE con struttura:
        /phase_space  (N, 7) float32   [x_cm, y_cm, z_cm, dx, dy, dz, E_MeV]
        /metadata     attrs            [model, checkpoint, config...]
"""

import argparse
import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser(
        description="Generazione phase space con modello addestrato",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",  type=str, required=True,
                   help="Path al file .pt del modello salvato")
    p.add_argument("--model",       choices=["nsf", "cfm", "gan"], required=True)
    p.add_argument("--stats_path",  type=str, default=None,
                   help="Path al JSON delle statistiche di normalizzazione. "
                        "Se None, cerca nella stessa cartella del checkpoint.")
    p.add_argument("--out",         type=str, default="generated_ps.h5",
                   help="Path di output per il file HDF5")
    p.add_argument("--n_samples",   type=int, default=1_000_000)
    p.add_argument("--batch_size",  type=int, default=50_000,
                   help="Campioni per batch (ridurre se OOM)")
    p.add_argument("--device",      type=str, default="auto")

    # Condizioni del fascio (per modelli condizionati)
    p.add_argument("--E_nom",  type=float, default=6.0,  help="Energia fascio [MeV]")
    p.add_argument("--jaw_x",  type=float, default=5.0,  help="Semi-jaw X [cm]")
    p.add_argument("--jaw_y",  type=float, default=5.0,  help="Semi-jaw Y [cm]")
    p.add_argument("--conditional", action="store_true",
                   help="Usa condizionamento (richiede checkpoint condizionato)")

    # CFM-specific
    p.add_argument("--fast",        action="store_true",
                   help="[CFM] Usa Euler (10 step) invece di DOPRI5 per produzione")
    p.add_argument("--n_ode_steps", type=int, default=50,
                   help="[CFM] Step ODE per DOPRI5")

    # Validazione rapida
    p.add_argument("--validate",    action="store_true",
                   help="Calcola W1 e MMD sui campioni generati vs un campione sintetico")
    p.add_argument("--n_val",       type=int, default=50_000,
                   help="Campioni per la validazione rapida")
    return p.parse_args()


def load_model(args, device):
    """Carica il modello dal checkpoint."""
    import torch

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Inferisci dimensioni dal checkpoint
    if args.model == "nsf":
        from models.nsf import PhaseSpaceNSF
        # Tenta di inferire i parametri dallo state dict
        sd = ckpt["model"]
        # Cerca il context encoder per capire se è condizionato
        cond_dim = 3 if any("cond_encoder" in k for k in sd.keys()) else 0
        model = PhaseSpaceNSF(dim=6, cond_dim=cond_dim)
        model.load_state_dict(sd)

    elif args.model == "cfm":
        from models.cfm import PhaseSpaceCFM
        sd = ckpt["model"]
        cond_dim = 3 if any("cond_embed" in k for k in sd.keys()) else 0
        # Inferisci hidden_dim dalla prima proiezione
        in_proj_shape = sd.get("velocity_net.input_proj.weight", None)
        hidden_dim = in_proj_shape.shape[0] if in_proj_shape is not None else 256
        model = PhaseSpaceCFM(dim=6, cond_dim=cond_dim, n_layers=8, hidden_dim=512)
        model.load_state_dict(sd)

    elif args.model == "gan":
        from models.gan import PhaseSpaceGenerator
        sd = ckpt["generator"]
        cond_dim = 3 if any("cond" in k for k in sd.keys()) else 0
        model = PhaseSpaceGenerator(cond_dim=cond_dim)
        model.load_state_dict(sd)

    model = model.to(device)
    model.eval()
    print(f"  Modello caricato da: {args.checkpoint}")
    print(f"  Condizionamento: {'sì (cond_dim=3)' if cond_dim > 0 else 'no'}")
    return model, cond_dim


def build_condition_tensor(args, n_samples, cond_stats, device):
    """Costruisce il tensore di condizione normalizzato."""
    import torch
    c_raw = np.array([[args.E_nom, args.jaw_x, args.jaw_y]], dtype=np.float32)
    c_raw = np.tile(c_raw, (n_samples, 1))

    # Normalizza con le stesse statistiche del training
    mu    = np.array(cond_stats["mu"],    dtype=np.float32)
    sigma = np.array(cond_stats["sigma"], dtype=np.float32)
    c_norm = (c_raw - mu) / sigma

    return torch.from_numpy(c_norm).to(device)


def generate_samples(model, args, cond_stats, stats, device):
    """
    Genera n_samples in batch e li denormalizza.
    Restituisce array (N, 7) nel sistema fisico originale [cm, adim, MeV].
    """
    import torch
    from data.synthetic_linac import denormalize_phase_space

    all_samples = []
    n_remaining = args.n_samples
    t0 = time.time()

    print(f"\n  Generazione {args.n_samples:,} campioni "
          f"(batch_size={args.batch_size:,})...")

    while n_remaining > 0:
        n_batch = min(args.batch_size, n_remaining)

        # Condizione (se applicabile)
        c = None
        if args.conditional and cond_stats is not None:
            c = build_condition_tensor(args, n_batch, cond_stats, device)

        # Campionamento
        with torch.no_grad():
            if args.model == "nsf":
                s = model.sample(n_batch, c=c).cpu().numpy()

            elif args.model == "cfm":
                if args.fast:
                    s = model.sample_fast(n_batch, c=c, n_steps=10).cpu().numpy()
                else:
                    s = model.sample(n_batch, c=c,
                                     n_steps=args.n_ode_steps).cpu().numpy()

            elif args.model == "gan":
                z = torch.randn(n_batch, model.latent_dim, device=device)
                s = model(z, c).cpu().numpy()

        all_samples.append(s)
        n_remaining -= n_batch

        # Progress
        n_done = args.n_samples - n_remaining
        elapsed = time.time() - t0
        rate    = n_done / elapsed if elapsed > 0 else 0
        print(f"  {n_done:>10,} / {args.n_samples:,} "
              f"({100*n_done/args.n_samples:.1f}%)  "
              f"{rate:,.0f} samples/s", end="\r")

    print()
    elapsed = time.time() - t0
    print(f"  Generazione completata in {elapsed:.1f}s "
          f"({args.n_samples/elapsed:,.0f} samples/s)")

    # Concatena e denormalizza
    samples_norm = np.concatenate(all_samples, axis=0)
    samples_phys = denormalize_phase_space(samples_norm, stats)

    return samples_phys


def validate_samples(generated, args):
    """Validazione rapida: confronto con campione sintetico della stessa config."""
    from data.synthetic_linac import generate_phase_space
    from evaluate import wasserstein1_marginals, mmd_rbf, separability_score

    print(f"\n  Validazione rapida ({args.n_val:,} campioni)...")
    reference = generate_phase_space(
        args.n_val, E_nom=args.E_nom,
        jaw_x=args.jaw_x, jaw_y=args.jaw_y, seed=999
    )

    n = min(args.n_val, len(generated))
    rng = np.random.default_rng(0)
    gen_sub = generated[rng.choice(len(generated), n, replace=False)]

    w1  = wasserstein1_marginals(reference, gen_sub)
    mmd = mmd_rbf(reference, gen_sub, n_subsample=min(3000, n))
    sep = separability_score(reference, gen_sub, n_subsample=min(5000, n))

    col_names = ["x", "y", "dx", "dy", "dz", "E"]
    print(f"\n  {'Canale':<6} {'W1':>10}")
    print(f"  {'-'*18}")
    for col in col_names:
        print(f"  {col:<6} {w1[col]:>10.6f}")
    print(f"  {'mean':<6} {w1['mean']:>10.6f}")
    print(f"\n  MMD^2        = {mmd:.6f}  (sqrt = {np.sqrt(mmd):.6f})")
    print(f"  Separability = {sep['accuracy']:.4f} ± {sep['std']:.4f}")
    print(f"  (sep=0.50 ottimo, sep=1.00 fallimento)")

    return {"w1": w1, "mmd": mmd, "separability": sep}


def main():
    args = parse_args()

    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch richiesto per la generazione.")
        print("Installare: pip install torch")
        sys.exit(1)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"\n  Device: {device}")

    # Statistiche di normalizzazione
    ckpt_dir = Path(args.checkpoint).parent
    stats_path = args.stats_path or str(ckpt_dir / "normalization_stats.json")
    cond_stats_path = str(ckpt_dir / "condition_stats.json")

    if not Path(stats_path).exists():
        print(f"[ERROR] File statistiche non trovato: {stats_path}")
        print("Specificare con --stats_path oppure usare la cartella del checkpoint.")
        sys.exit(1)

    with open(stats_path) as f:
        stats = json.load(f)
    print(f"  Statistiche caricate da: {stats_path}")

    cond_stats = None
    if Path(cond_stats_path).exists():
        with open(cond_stats_path) as f:
            cond_stats = json.load(f)
        print(f"  Statistiche condizioni caricate da: {cond_stats_path}")

    # Carica modello
    model, cond_dim = load_model(args, device)

    # Genera
    generated = generate_samples(model, args, cond_stats, stats, device)

    print(f"\n  Campioni generati: {generated.shape}")
    print(f"  Statistiche fisiche:")
    col_names_full = ["x [cm]", "y [cm]", "z [cm]", "dx", "dy", "dz", "E [MeV]"]
    for i, name in enumerate(col_names_full):
        col = generated[:, i]
        print(f"    {name:>10s}: "
              f"mu={col.mean():+7.4f}  "
              f"σ={col.std():7.4f}  "
              f"[{col.min():+8.4f}, {col.max():+8.4f}]")

    # Verifica vincolo fisico
    d_norms = np.linalg.norm(generated[:, 3:6], axis=1)
    max_dev = np.abs(d_norms - 1).max()
    print(f"\n  ||d||=1 max deviation: {max_dev:.2e} "
          f"{'✓' if max_dev < 1e-4 else '✗ ATTENZIONE'}")

    # Validazione opzionale
    val_results = None
    if args.validate:
        val_results = validate_samples(generated, args)

    # Salva HDF5
    from data.synthetic_linac import save_phase_space_hdf5
    metadata = {
        "model":      args.model,
        "checkpoint": args.checkpoint,
        "E_nom":      str(args.E_nom),
        "jaw_x":      str(args.jaw_x),
        "jaw_y":      str(args.jaw_y),
        "n_samples":  str(args.n_samples),
        "device":     device,
    }
    if val_results:
        metadata["val_w1_mean"] = str(val_results["w1"]["mean"])
        metadata["val_sep"]     = str(val_results["separability"]["accuracy"])

    save_phase_space_hdf5(generated, None, args.out, metadata=metadata)
    print(f"\n  ✓ Phase space salvato: {args.out}")
    print(f"\n  Uso in GATE (Python API):")
    print(f"    source = sim.add_source('PhaseSpaceSource', 'beam')")
    print(f"    source.phsp_file = '{args.out}'")
    print(f"    source.particle = 'gamma'")


if __name__ == "__main__":
    main()
