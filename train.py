"""
train.py
========
Script di training unificato per tutti i modelli.

Uso:
    # Genera dati sintetici e addestra NSF (non condizionato)
    python train.py --model nsf --n_samples 500000 --epochs 100

    # Addestra CFM condizionato su multi-config
    python train.py --model cfm --conditional --epochs 200

    # Addestra baseline GAN
    python train.py --model gan --epochs 300

    # Usa dati GATE reali invece dei sintetici
    python train.py --model nsf --data_path /path/to/ps.h5

Opzioni principali:
    --model        : nsf | cfm | gan
    --conditional  : attiva il condizionamento su c = [E_nom, jaw_x, jaw_y]
    --n_samples    : campioni sintetici da generare (default: 500000)
    --batch_size   : batch size (default: 2048)
    --epochs       : epoche di training (default: 200)
    --lr           : learning rate (default: 1e-3)
    --device       : cuda | cpu (default: auto-detect)
    --output_dir   : directory per checkpoints e plots (default: outputs/)
"""

import argparse
import os
import sys
import json
import time
import numpy as np
from pathlib import Path

# ─── Aggiungi la root al path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from data.synthetic_linac import (
    generate_phase_space,
    generate_multi_condition_dataset,
    save_phase_space_hdf5,
    normalize_phase_space,
    denormalize_phase_space,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Training modelli generativi per phase space MC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Modello
    p.add_argument("--model", choices=["nsf", "cfm", "gan"], default="cfm",
                   help="Modello generativo da addestrare")
    p.add_argument("--conditional", action="store_true",
                   help="Usa condizionamento su parametri del fascio")

    # Dati
    p.add_argument("--data_path", type=str, default=None,
                   help="Path a file HDF5 con phase space GATE reale. "
                        "Se None, generates dati sintetici.")
    p.add_argument("--n_samples",  type=int, default=500_000,
                   help="Campioni da generare (solo per dati sintetici)")
    p.add_argument("--E_nom",  type=float, default=6.0,  help="Energia fascio [MeV]")
    p.add_argument("--jaw_x",  type=float, default=5.0,  help="Semi-apertura jaw X [cm]")
    p.add_argument("--jaw_y",  type=float, default=5.0,  help="Semi-apertura jaw Y [cm]")

    # Training
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch_size",  type=int,   default=2048)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--device",      type=str,   default="auto")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--val_every",   type=int,   default=5,
                   help="Calcola val loss ogni N epoche")
    p.add_argument("--save_every",  type=int,   default=20,
                   help="Salva checkpoint ogni N epoche")
    p.add_argument("--n_critic", type=int, default=5,
                   help="[GAN] Aggiornamenti del critic per ogni update del generatore (Sarrut: 4)")
    p.add_argument("--gan_hidden_dims", type=int, nargs="+", default=[256, 512, 512, 256],
                   help="[GAN] Dimensioni degli hidden layers per G e D (Sarrut: 400 400 400)")
    # NSF-specific
    p.add_argument("--spherical",    action="store_true",
                   help="Reparametrizzazione sferica (theta,phi) — per dati reali IAEA")
    p.add_argument("--rank_spatial", action="store_true",
                   help="Rank Transform su x,y (Copula) — risolve bordi netti jaw")
    p.add_argument("--n_transforms", type=int, default=6,
                   help="[NSF] Numero di coupling transforms")
    p.add_argument("--n_bins",        type=int, default=8,
                   help="[NSF] Numero di bin delle spline")
    p.add_argument("--hidden_dim",    type=int, default=128,
                   help="Dimensione hidden layers")
    p.add_argument("--tail_bound",    type=float, default=5.0,
                   help="[NSF] Limite della spline in unita di sigma")
    # CFM-specific
    p.add_argument("--n_layers",      type=int, default=4,
                   help="[CFM] Numero di layer residuali nella velocity net")
    p.add_argument("--n_ode_steps",   type=int, default=100,
                   help="[CFM] Step ODE per il campionamento in validazione")

    # Output
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--run_name",   type=str, default=None,
                   help="Nome del run (default: model_timestamp)")

    return p.parse_args()


def setup_output_dir(args) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = args.run_name or f"{args.model}_{ts}"
    out = Path(args.output_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    # Salva config
    with open(out / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    return out


def prepare_data(args, out_dir: Path):
    """
    Prepara i dati di training (sintetici o reali).
    Restituisce:
        (ps_train, ps_val, ps_test,
         c_train, c_val, c_test,   -- None se non condizionato
         stats)                    -- statistiche di normalizzazione
    """
    print("\n" + "="*55)
    print("  Preparazione dati")
    print("="*55)

    if args.data_path is not None:
        # ── Dati GATE reali ────────────────────────────────────────────────
        print(f"  Caricamento da: {args.data_path}")
        from data.synthetic_linac import load_phase_space_hdf5
        ps, conditions = load_phase_space_hdf5(args.data_path)
        print(f"  Caricati {len(ps):,} vettori di phase space")

    elif args.conditional:
        # ── Dataset multi-condizione sintetico ────────────────────────────
        print("  Generazione dataset multi-condizione sintetico...")
        from data.synthetic_linac import DEFAULT_CONFIGS
        n_per = args.n_samples // len(DEFAULT_CONFIGS)
        ps, conditions, label_map = generate_multi_condition_dataset(
            n_per_config=n_per,
            seed=args.seed,
        )
        print(f"  Generati {len(ps):,} vettori | {len(DEFAULT_CONFIGS)} configurazioni")
        print(f"  Config: {list(label_map.keys())}")

    else:
        # ── Phase space singola configurazione ────────────────────────────
        print(f"  Generazione {args.n_samples:,} campioni sintetici...")
        print(f"  Config: E={args.E_nom}MV, jaw=({args.jaw_x}×{args.jaw_y})cm")
        ps = generate_phase_space(
            n_samples=args.n_samples,
            E_nom=args.E_nom,
            jaw_x=args.jaw_x,
            jaw_y=args.jaw_y,
            seed=args.seed,
        )
        conditions = None

    # Salva raw data per reference
    save_phase_space_hdf5(
        ps, conditions, str(out_dir / "data_raw.h5"),
        metadata=vars(args)
    )

    # Normalizzazione
    use_spherical = getattr(args, "spherical", False)
    use_rank      = getattr(args, "rank_spatial", False)
    # Chiamata aggiornata con i nuovi parametri per la Copula spaziale
    ps_norm, stats = normalize_phase_space(
        ps, 
        spherical=use_spherical, 
        rank_spatial=use_rank, 
        stats_dir=str(out_dir)
    )
    if use_spherical:
        print(f"  Reparametrizzazione: (dx,dy,dz) -> (theta,phi)")

    z_vals = ps[:, 2] if ps.shape[1] == 7 else np.zeros(1)
    if z_vals.std() < 1e-3:
        stats["z_const"] = float(z_vals.mean())

    with open(out_dir / "normalization_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Statistiche di normalizzazione salvate")

    # Normalizza anche le condizioni
    if conditions is not None:
        from data.dataset import normalize_conditions

        c_norm, c_stats = normalize_conditions(conditions)
        with open(out_dir / "condition_stats.json", "w") as f:
            json.dump(c_stats, f, indent=2)
    else:
        c_norm = None

    # Split 70/15/15
    n = len(ps_norm)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    tr_idx  = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    te_idx  = perm[n_train + n_val:]

    ps_train = ps_norm[tr_idx]
    ps_val   = ps_norm[val_idx]
    ps_test  = ps_norm[te_idx]

    if c_norm is not None:
        c_train = c_norm[tr_idx]
        c_val   = c_norm[val_idx]
        c_test  = c_norm[te_idx]
    else:
        c_train = c_val = c_test = None

    print(f"  Split: train={len(ps_train):,} | val={len(ps_val):,} | test={len(ps_test):,}")

    return ps_train, ps_val, ps_test, c_train, c_val, c_test, stats, ps


def run_training(args):
    """Entry point principale del training."""

    # ── Import PyTorch ─────────────────────────────────────────────────────
    try:
        import torch
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("[ERROR] PyTorch non disponibile.")
        print("Installare con: pip install torch --extra-index-url https://download.pytorch.org/whl/cpu")
        sys.exit(1)

    # ── Setup ──────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    out_dir = setup_output_dir(args)

    print(f"\n{'='*55}")
    print(f"  Phase Space Generative Model Training")
    print(f"  Modello:  {args.model.upper()}")
    print(f"  Device:   {device}")
    print(f"  Output:   {out_dir}")
    print(f"{'='*55}")

    # ── Dati ───────────────────────────────────────────────────────────────
    ps_tr, ps_val, ps_te, c_tr, c_val, c_te, stats, ps_raw = prepare_data(args, out_dir)

    cond_dim = c_tr.shape[1] if c_tr is not None else 0

    # Converti in tensori
    def to_tensor(arr):
        return torch.from_numpy(arr).float() if arr is not None else None

    X_tr  = to_tensor(ps_tr);   C_tr  = to_tensor(c_tr)
    X_val = to_tensor(ps_val);  C_val = to_tensor(c_val)
    X_te  = to_tensor(ps_te);   C_te  = to_tensor(c_te)

    # DataLoader per il training
    if cond_dim > 0:
        train_ds = TensorDataset(X_tr, C_tr)
    else:
        train_ds = TensorDataset(X_tr)

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device == "cuda"),
    )

    # ── Costruzione modello ────────────────────────────────────────────────
    print(f"\n  Costruzione modello {args.model.upper()}...")
    if args.model == "nsf":
        from models.nsf import PhaseSpaceNSF, NSFTrainer
        _dim = 5 if getattr(args, "spherical", False) else 6
        model   = PhaseSpaceNSF(
            dim=_dim, n_transforms=args.n_transforms,
            hidden_dim=args.hidden_dim, n_bins=args.n_bins,
            cond_dim=cond_dim,
            tail_bound=args.tail_bound
        )
        trainer = NSFTrainer(model, device=device, lr=args.lr)

    elif args.model == "cfm":
        from models.cfm import PhaseSpaceCFM, CFMTrainer
        _dim_cfm = 5 if getattr(args, "spherical", False) else 6
        model   = PhaseSpaceCFM(
            dim=_dim_cfm, cond_dim=cond_dim,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        )
        trainer = CFMTrainer(model, device=device, lr=args.lr, epochs=args.epochs)

    elif args.model == "gan":
        from models.gan import PhaseSpaceGenerator, PhaseSpaceCritic, WGANGPTrainer
        # Estrarre la dimensione reale direttamente dalla matrice dei dati previene qualsiasi NameError
        _dim_gan = ps_tr.shape[1]
        G = PhaseSpaceGenerator(cond_dim=cond_dim, hidden_dims=args.gan_hidden_dims, output_dim=_dim_gan)
        D = PhaseSpaceCritic(cond_dim=cond_dim,    hidden_dims=args.gan_hidden_dims, input_dim=_dim_gan)
        trainer = WGANGPTrainer(G, D, device=device, lr=args.lr, n_critic=args.n_critic)
        model = G

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametri: {n_params:,}")
    if args.model == "gan":
        print(f"  n_critic: {args.n_critic} | gan_hidden_dims: {args.gan_hidden_dims}")

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\n  Inizio training ({args.epochs} epoche)...")
    print(f"  Batch size: {args.batch_size} | Steps/epoca: {len(loader)}")
    print(f"{'─'*55}")

    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # ── Training epoch ────────────────────────────────────────────────
        model.train()
        losses = []
        for batch in loader:
            if cond_dim > 0:
                s_b, c_b = batch
            else:
                s_b = batch[0]
                c_b = None

            if args.model in ("nsf", "cfm"):
                loss = trainer.train_step(s_b, c_b)
                losses.append(loss)
            else:  # GAN
                metrics = trainer.train_step(s_b, c_b)
                losses.append(metrics["w_dist"])

        train_loss = np.mean(losses)

        if epoch % args.val_every == 0 or epoch == args.epochs:
            if args.model in ("nsf", "cfm"):
                val_loss = trainer.val_step(X_val, C_val)
                if args.model == "cfm":  # CosineAnnealingLR stepped per epoch
                    trainer.scheduler.step()
                trainer.history["train_nll" if args.model == "nsf" else "train_loss"].append(train_loss)
                trainer.history["val_nll" if args.model == "nsf" else "val_loss"].append(val_loss)
                trainer.history["lr"].append(trainer.opt.param_groups[0]["lr"])
            else:
                val_loss = train_loss  # GAN non ha val loss classica

            # Estrattore di Learning Rate universale e sicuro per NSF, CFM e GAN
            if hasattr(trainer, 'opt'):
                current_lr = trainer.opt.param_groups[0]['lr']
            elif hasattr(trainer, 'opt_G'):
                current_lr = trainer.opt_G.param_groups[0]['lr']
            elif hasattr(trainer, 'optimizer_G'):
                current_lr = trainer.optimizer_G.param_groups[0]['lr']
            else:
                current_lr = args.lr

            elapsed = time.time() - t_start
            print(f" Epoch {epoch:>4d}/{args.epochs} | "
                  f"train: {train_loss:>9.5f} | "
                  f"val: {val_loss:>9.5f} | "
                  f"lr: {current_lr:.2e} | "
                  f"{elapsed:.0f}s")

            # Salva best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save(str(out_dir / "best_model.pt"))

        # ── Checkpoint periodico ──────────────────────────────────────────
        if epoch % args.save_every == 0:
            trainer.save(str(out_dir / f"checkpoint_ep{epoch:04d}.pt"))

    # ── Valutazione finale ────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  Valutazione finale sul test set...")
    print(f"{'='*55}")

    # Carica il best model (non l'ultimo stato dell'ultima epoca)
    best_ckpt = str(out_dir / "best_model.pt")
    print(f"  Caricamento best model: val={best_val_loss:.5f}")
    trainer.load(best_ckpt)

    # Genera campioni con il modello
    model.eval()
    n_gen = len(ps_te)
    chunk_size = 250000  # 250k campioni alla volta entrano comodi nei 16GB di VRAM
    gen_norm_list = [] 
    print(f"\n  Generazione finale sul test set in corso ({n_gen:,} campioni in chunk da {chunk_size:,})...")
    with torch.no_grad():
        for i in range(0, n_gen, chunk_size):
            end_idx = min(i + chunk_size, n_gen)
            size = end_idx - i
            # Seleziona lo spezzone di condizioni (se presenti)
            c_chunk = C_te[i:end_idx].to(device) if C_te is not None else None

            if args.model == "nsf":
                chunk_out = model.sample(size, c=c_chunk).cpu().numpy()
            elif args.model == "cfm":
                chunk_out = model.sample(size, c=c_chunk, n_steps=args.n_ode_steps).cpu().numpy()
            else:  # GAN
                chunk_out = model.sample(size, c=c_chunk, device=device).cpu().numpy()

            gen_norm_list.append(chunk_out)
            print(f"  Avanzamento: {end_idx:,} / {n_gen:,} ({100*end_idx/n_gen:.1f}%)", end="\r")
    print()
    # Concatena tutti i blocchi generati in un'unica matrice finale
    gen_norm = np.concatenate(gen_norm_list, axis=0)
    # Denormalizza
    gen_raw = denormalize_phase_space(gen_norm, stats)
    real_test_raw = denormalize_phase_space(ps_te, stats)

    # Metriche
    from evaluate import evaluate_model
    report = evaluate_model(
        real_test_raw, gen_raw,
        model_name=args.model,
        output_dir=str(out_dir / "eval"),
    )

    # Salva campioni generati per uso downstream (GATE)
    save_phase_space_hdf5(
        gen_raw, None, str(out_dir / "generated_ps.h5"),
        metadata={"model": args.model, "n_samples": str(n_gen)}
    )
    print(f"\n  Campioni generati salvati: {out_dir}/generated_ps.h5")
    print(f"  Uso in GATE: impostare come sorgente il file HDF5")
    print(f"\n  Training completato! Output in: {out_dir}")

    return report


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
