"""
eval_conditional_real.py
=========================
Valutazione condizionale per-energia di qualsiasi modello (CFM, NSF, GAN).

Carica un modello condizionale dal checkpoint, genera campioni per ciascuna
energia e li confronta con il test set reale della stessa energia.

Uso:
    python eval_conditional_real.py \\
        --model cfm --run_dir outputs/cfm_conditional_6mv_10mv \\
        --data_6mv data/elekta_6mv_train_20M.h5 \\
        --data_10mv data/elekta_10mv.h5

    python eval_conditional_real.py \\
        --model nsf --run_dir outputs/nsf_conditional_6mv_10mv

    python eval_conditional_real.py \\
        --model gan --run_dir outputs/gan_conditional_6mv_10mv
"""

import argparse
import json
import sys
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.synthetic_linac import (
    load_phase_space_hdf5, denormalize_phase_space,
    generate_phase_space,
)
from evaluate import (
    evaluate_model, wasserstein1_marginals, mmd_rbf,
    separability_score, plot_marginals,
)


CHUNK_SIZE = 250_000
# Configurazioni di default per la valutazione
DEFAULT_EVAL_CONFIGS = {
    "6MV": {"E_nom": 6.0, "jaw_x": 5.0, "jaw_y": 5.0},
    "10MV": {"E_nom": 10.0, "jaw_x": 5.0, "jaw_y": 5.0},
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Valutazione condizionale per-energia",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=["nsf", "cfm", "gan"], required=True)
    p.add_argument("--run_dir", type=str, required=True,
                   help="Cartella del run di training")
    p.add_argument("--data_6mv", type=str, default=None,
                   help="Path al file HDF5 del 6MV per evaluation. "
                        "Se None, usa dati sintetici.")
    p.add_argument("--data_10mv", type=str, default=None,
                   help="Path al file HDF5 del 10MV per evaluation. "
                        "Se None, usa dati sintetici.")
    p.add_argument("--n_samples", type=int, default=50_000,
                   help="Campioni da generare per configurazione")
    p.add_argument("--n_ode_steps", type=int, default=None,
                   help="[CFM] Step ODE inferenza. Default: da config.json")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def load_run(run_dir: Path):
    """Carica config.json e normalization_stats.json dal run."""
    cfg_path = run_dir / "config.json"
    stats_path = run_dir / "normalization_stats.json"
    cond_stats_path = run_dir / "condition_stats.json"

    if not cfg_path.exists():
        print(f"[ERROR] {cfg_path} non trovato.")
        sys.exit(1)
    if not stats_path.exists():
        print(f"[ERROR] {stats_path} non trovato.")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = json.load(f)
    with open(stats_path) as f:
        stats = json.load(f)

    cond_stats = None
    if cond_stats_path.exists():
        with open(cond_stats_path) as f:
            cond_stats = json.load(f)

    return cfg, stats, cond_stats


def build_model(model_type: str, cfg: dict, stats: dict, run_dir: Path, device: str):
    """Costruisce e carica il modello dal config.json del run."""
    col_names = stats.get("col_names", ["x", "y", "dx", "dy", "dz", "E"])
    dim = len(col_names)
    cond_dim = 3 if cfg.get("conditional", False) else 0

    if model_type == "nsf":
        from models.nsf import PhaseSpaceNSF, NSFTrainer
        model = PhaseSpaceNSF(
            dim=dim,
            n_transforms=cfg.get("n_transforms", 6),
            hidden_dim=cfg.get("hidden_dim", 128),
            n_bins=cfg.get("n_bins", 8),
            tail_bound=cfg.get("tail_bound", 5.0),
            cond_dim=cond_dim,
        )
        trainer = NSFTrainer(model, device=device, lr=cfg.get("lr", 1e-3))
        trainer.load(str(run_dir / "best_model.pt"))

    elif model_type == "cfm":
        from models.cfm import PhaseSpaceCFM, CFMTrainer
        model = PhaseSpaceCFM(
            dim=dim,
            cond_dim=cond_dim,
            hidden_dim=cfg.get("hidden_dim", 256),
            n_layers=cfg.get("n_layers", 4),
        )
        trainer = CFMTrainer(model, device=device, lr=cfg.get("lr", 1e-3))
        trainer.load(str(run_dir / "best_model.pt"))

    elif model_type == "gan":
        from models.gan import PhaseSpaceGenerator, PhaseSpaceCritic, WGANGPTrainer
        hidden_dims = cfg.get("gan_hidden_dims", [256, 512, 512, 256])
        G = PhaseSpaceGenerator(cond_dim=cond_dim, hidden_dims=hidden_dims, output_dim=dim)
        D = PhaseSpaceCritic(cond_dim=cond_dim, hidden_dims=hidden_dims, input_dim=dim)
        trainer = WGANGPTrainer(G, D, device=device, lr=cfg.get("lr", 1e-4),
                                n_critic=cfg.get("n_critic", 5))
        trainer.load(str(run_dir / "best_model.pt"))
        model = G

    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Modello {model_type.upper()} caricato: {n_params:,} parametri")
    return model


def generate_conditional(model, model_type, c_norm, n_samples, device, n_ode_steps=100):
    """Genera campioni condizionati in chunk."""
    all_samples = []
    n_done = 0

    with torch.no_grad():
        while n_done < n_samples:
            n_chunk = min(CHUNK_SIZE, n_samples - n_done)
            c_chunk = c_norm[:n_chunk].to(device) if c_norm is not None else None

            if model_type == "nsf":
                s = model.sample(1, c=c_chunk).cpu().numpy()
                if len(s.shape) == 3: s = s.squeeze(1)
            elif model_type == "cfm":
                try:
                    s = model.sample(n_chunk, c=c_chunk, n_steps=n_ode_steps).cpu().numpy()
                except Exception:
                    s = model.sample_fast(n_chunk, c=c_chunk, n_steps=50).cpu().numpy()
            elif model_type == "gan":
                s = model.sample(n_chunk, c=c_chunk, device=device).cpu().numpy()

            all_samples.append(s)
            n_done += n_chunk
            print(f"\r    Generati {n_done:,} / {n_samples:,}", end="")

    print()
    return np.concatenate(all_samples, axis=0)


def get_reference_data(energy_key: str, data_path: str = None, n_samples: int = 50_000):
    """
    Carica il reference data per una data energia.
    Se data_path è fornito, usa quello; altrimenti genera sinteticamente.
    """
    cfg = DEFAULT_EVAL_CONFIGS[energy_key]

    if data_path and Path(data_path).exists():
        ps, _ = load_phase_space_hdf5(data_path, max_samples=n_samples)
        print(f"    Reference caricato da: {data_path} ({len(ps):,} campioni)")
        return ps

    print(f"    [INFO] Nessun file eval per {energy_key}. Uso dati sintetici.")
    ps = generate_phase_space(
        n_samples=n_samples,
        E_nom=cfg["E_nom"],
        jaw_x=cfg["jaw_x"],
        jaw_y=cfg["jaw_y"],
        seed=999,
    )
    return ps


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    print(f"\n{'='*60}")
    print(f"  Valutazione condizionale: {run_dir.name}  [{args.model.upper()}]")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Carica config e stats del training
    cfg, stats, cond_stats = load_run(run_dir)

    if cond_stats is None:
        print("[ERROR] condition_stats.json non trovato. "
              "Il modello non è stato addestrato in condizionale?")
        sys.exit(1)

    # Costruisci il modello
    model = build_model(args.model, cfg, stats, run_dir, device)

    # Statistiche condizioni per normalizzazione
    mu_c = np.array(cond_stats["mu"], dtype=np.float32)
    sig_c = np.array(cond_stats["sigma"], dtype=np.float32)

    n_ode_steps = args.n_ode_steps or cfg.get("n_ode_steps", 100)

    # Data paths per energia
    data_paths = {"6MV": args.data_6mv, "10MV": args.data_10mv}

    output_dir = run_dir / "eval_conditional"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    summary_rows = []

    for energy_key, eval_cfg in DEFAULT_EVAL_CONFIGS.items():
        E_nom = eval_cfg["E_nom"]
        jaw_x = eval_cfg["jaw_x"]
        jaw_y = eval_cfg["jaw_y"]

        print(f"\n{'━'*60}")
        print(f"  [{energy_key}] E_nom={E_nom} MeV, jaw=({jaw_x}×{jaw_y}) cm")
        print(f"{'━'*60}")

        # Reference data
        ref_data = get_reference_data(energy_key, data_paths.get(energy_key), args.n_samples)

        # Genera campioni condizionati
        c_raw = np.array([[E_nom, jaw_x, jaw_y]], dtype=np.float32)
        c_norm = (c_raw - mu_c) / sig_c
        c_tensor = torch.from_numpy(
            np.tile(c_norm, (args.n_samples, 1))
        ).float()

        print(f"    Generazione {args.n_samples:,} campioni...")
        t0 = time.time()
        gen_norm = generate_conditional(
            model, args.model, c_tensor, args.n_samples, device, n_ode_steps
        )
        elapsed = time.time() - t0
        print(f"    Generati in {elapsed:.1f}s")

        # Denormalizza
        gen_phys = denormalize_phase_space(gen_norm, stats)

        # Metriche
        w1 = wasserstein1_marginals(ref_data, gen_phys)
        mmd = mmd_rbf(ref_data, gen_phys, n_subsample=min(10_000, args.n_samples))
        sep = separability_score(ref_data, gen_phys, n_subsample=min(10_000, args.n_samples))

        print(f"    W1_mean={w1['mean']:.6f}  W1_E={w1['E']:.6f}  "
              f"MMD²={mmd:.6f}  Sep={sep['accuracy']:.4f}")

        # Leakage check: particelle con E > E_nom
        e_gen = gen_phys[:, 6]
        leakage_frac = (e_gen > E_nom * 1.05).mean()
        print(f"    Cross-energy leakage (E > {E_nom*1.05:.1f} MeV): {leakage_frac:.4%}")

        all_results[energy_key] = {
            "w1": w1, "mmd": mmd, "separability": sep,
            "leakage_frac": float(leakage_frac),
            "config": eval_cfg,
        }
        summary_rows.append((energy_key, w1["mean"], w1["E"], mmd,
                             sep["accuracy"], leakage_frac))

        # Plot marginali per questa energia
        plot_marginals(
            ref_data, {f"{args.model.upper()} ({energy_key})": gen_phys},
            str(output_dir / f"marginals_{energy_key}.png"),
        )

    # Tabella riassuntiva
    print(f"\n{'='*75}")
    print(f"  {'Config':<8} {'W1_mean':>10} {'W1_E':>10} {'MMD²':>12} {'Sep':>8} {'Leakage':>10}")
    print(f"  {'-'*67}")
    for row in summary_rows:
        print(f"  {row[0]:<8} {row[1]:>10.6f} {row[2]:>10.6f} "
              f"{row[3]:>12.6f} {row[4]:>8.4f} {row[5]:>10.4%}")

    mean_w1 = np.mean([r[1] for r in summary_rows])
    mean_sep = np.mean([r[4] for r in summary_rows])
    print(f"  {'MEDIA':<8} {mean_w1:>10.6f} {'':>10} {'':>12} {mean_sep:>8.4f}")
    print(f"{'='*75}")

    # Salva report
    report_path = output_dir / f"{args.model}_conditional_eval.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\n  ✓ Report salvato: {report_path}")


if __name__ == "__main__":
    main()
