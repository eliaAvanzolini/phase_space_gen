"""
eval_final.py
=============
Valutazione finale dei modelli (NSF / CFM / GAN) addestrati su 130M
campioni, testati sull'INTERO dataset di eval indipendente
(elekta_130mv_eval_completo.h5 — il PHSP2 di Sarrut).

PRINCIPIO CHIAVE (evita data leakage):
La denormalizzazione (incluse le rank-transform quantiles di NSF/CFM)
usa SEMPRE le statistiche calcolate durante il TRAINING
(normalization_stats.json del run). Non vengono MAI ricalcolate sul
file di eval. Gli iperparametri dell'architettura (n_transforms,
n_bins, hidden_dim, tail_bound, gan_hidden_dims, n_critic, ecc.)
sono letti dal config.json del run, MAI hardcoded.

Uso:
    python eval_final.py --run_dir outputs/cfm_130mln_rank_final  --model cfm
    python eval_final.py --run_dir outputs/nsf_130mln_rank_final  --model nsf
    python eval_final.py --run_dir outputs/gan_sarrut_replica_80k --model gan
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from data.synthetic_linac import load_phase_space_hdf5, denormalize_phase_space
from evaluate import evaluate_model


EVAL_DATA_PATH = "data/elekta_130mv_eval_completo.h5"
CHUNK_SIZE = 250_000  # campioni generati per chunk (sicuro su V100 16GB)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True,
                    help="Cartella del run di training (contiene best_model.pt, "
                         "normalization_stats.json, config.json)")
    p.add_argument("--model", choices=["nsf", "cfm", "gan"], required=True)
    p.add_argument("--data_path", type=str, default=EVAL_DATA_PATH)
    p.add_argument("--n_ode_steps", type=int, default=None,
                    help="[CFM] step ODE in inferenza. Default: legge da config.json, "
                         "altrimenti 100.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--run_name", type=str, default=None,
                    help="Nome per il report (default: <run_dir_name>_eval_full)")
    return p.parse_args()


def load_run_metadata(run_dir: Path):
    """Carica config.json e normalization_stats.json del run, con controlli."""
    cfg_path = run_dir / "config.json"
    stats_path = run_dir / "normalization_stats.json"

    if not cfg_path.exists():
        print(f"[ERROR] {cfg_path} non trovato.")
        sys.exit(1)
    if not stats_path.exists():
        print(f"[ERROR] {stats_path} non trovato. Impossibile denormalizzare "
              f"in modo coerente con il training.")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = json.load(f)
    with open(stats_path) as f:
        stats = json.load(f)

    return cfg, stats


def check_rank_quantiles(stats: dict, run_dir: Path):
    """Verifica che i file .npy dei quantili esistano se rank_spatial=True."""
    if not stats.get("rank_spatial", False):
        return
    q_dir = Path(stats.get("quantiles_dir", str(run_dir)))
    x_q = q_dir / "x_quantiles.npy"
    y_q = q_dir / "y_quantiles.npy"
    if not (x_q.exists() and y_q.exists()):
        print(f"[ERROR] rank_spatial=True ma quantili non trovati in {q_dir}")
        print(f"        Attesi: x_quantiles.npy, y_quantiles.npy")
        sys.exit(1)
    print(f"  Quantili rank transform: {q_dir} (dal TRAINING — nessun leakage)")


def build_and_load_nsf(cfg: dict, stats: dict, run_dir: Path, device: str):
    from models.nsf import PhaseSpaceNSF, NSFTrainer
    dim = len(stats.get("col_names", ["x", "y", "dx", "dy", "dz", "E"]))
    model = PhaseSpaceNSF(
        dim=dim,
        n_transforms=cfg["n_transforms"],
        hidden_dim=cfg["hidden_dim"],
        n_bins=cfg["n_bins"],
        tail_bound=cfg["tail_bound"],
        cond_dim=0,
    )
    trainer = NSFTrainer(model, device=device, lr=cfg["lr"])
    trainer.load(str(run_dir / "best_model.pt"))
    model.eval()
    print(f"  NSF ricostruito da config.json: n_transforms={cfg['n_transforms']}, "
          f"n_bins={cfg['n_bins']}, hidden_dim={cfg['hidden_dim']}, "
          f"tail_bound={cfg['tail_bound']}")
    return model


def build_and_load_cfm(cfg: dict, stats: dict, run_dir: Path, device: str):
    from models.cfm import PhaseSpaceCFM, CFMTrainer
    dim = len(stats.get("col_names", ["x", "y", "dx", "dy", "dz", "E"]))
    model = PhaseSpaceCFM(
        dim=dim,
        cond_dim=0,
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
    )
    trainer = CFMTrainer(model, device=device, lr=cfg["lr"])
    trainer.load(str(run_dir / "best_model.pt"))
    model.eval()
    print(f"  CFM ricostruito da config.json: hidden_dim={cfg['hidden_dim']}, "
          f"n_layers={cfg['n_layers']}")
    return model


def build_and_load_gan(cfg: dict, stats: dict, run_dir: Path, device: str):
    from models.gan import PhaseSpaceGenerator, PhaseSpaceCritic, WGANGPTrainer
    hidden_dims = cfg.get("gan_hidden_dims", [256, 512, 512, 256])
    G = PhaseSpaceGenerator(cond_dim=0, hidden_dims=hidden_dims)
    D = PhaseSpaceCritic(cond_dim=0, hidden_dims=hidden_dims)
    trainer = WGANGPTrainer(G, D, device=device, lr=cfg["lr"],
                             n_critic=cfg.get("n_critic", 5))
    trainer.load(str(run_dir / "best_model.pt"))
    G.eval()
    print(f"  GAN ricostruita da config.json: gan_hidden_dims={hidden_dims}, "
          f"n_critic={cfg.get('n_critic', 5)}")
    return G


def generate_in_chunks(sample_fn, n_total: int, chunk_size: int = CHUNK_SIZE):
    """Genera n_total campioni chiamando sample_fn(n_chunk) ripetutamente."""
    out = []
    n_done = 0
    with torch.no_grad():
        while n_done < n_total:
            n_chunk = min(chunk_size, n_total - n_done)
            g = sample_fn(n_chunk)
            out.append(g.cpu().numpy() if torch.is_tensor(g) else g)
            n_done += n_chunk
            print(f"\r  Avanzamento: {n_done:,} / {n_total:,} "
                  f"({100*n_done/n_total:.1f}%)", end="")
    print()
    return np.concatenate(out, axis=0)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    run_name = args.run_name or f"{run_dir.name}_eval_full"

    print(f"\n{'='*60}")
    print(f"  Valutazione finale: {run_dir.name}  [{args.model.upper()}]")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # ── 1. Carica config + stats DEL TRAINING ──────────────────────────────
    cfg, stats = load_run_metadata(run_dir)
    rank_spatial = stats.get("rank_spatial", False)
    print(f"  spherical={stats.get('spherical')} | rank_spatial={rank_spatial}")
    check_rank_quantiles(stats, run_dir)

    # ── 2. Carica il file di EVAL completo (dati fisici grezzi, MAI ──────
    #       ripassati per normalize_phase_space con stats=None) ───────────
    print(f"\n  Caricamento dataset di evaluation: {args.data_path}")
    ps_eval, _ = load_phase_space_hdf5(args.data_path)
    n_eval = len(ps_eval)
    print(f"  Campioni di eval: {n_eval:,}")
    real_raw = ps_eval.astype(np.float32)

    # ── 3. Costruisci e carica il modello dal config.json del run ──────────
    print(f"\n  Costruzione e caricamento modello {args.model.upper()}...")
    if args.model == "nsf":
        model = build_and_load_nsf(cfg, stats, run_dir, device)
        sample_fn = lambda n: model.sample(n)
    elif args.model == "cfm":
        model = build_and_load_cfm(cfg, stats, run_dir, device)
        n_steps = args.n_ode_steps or cfg.get("n_ode_steps", 100)
        print(f"  n_ode_steps inferenza: {n_steps}")
        sample_fn = lambda n: model.sample(n, n_steps=n_steps)
    elif args.model == "gan":
        model = build_and_load_gan(cfg, stats, run_dir, device)
        sample_fn = lambda n: model.sample(n, device=device)

    # ── 4. Genera n_eval campioni (in chunk per non saturare la VRAM) ──────
    print(f"\n  Generazione di {n_eval:,} campioni...")
    gen_norm = generate_in_chunks(sample_fn, n_eval)

    # ── 5. Denormalizza USANDO stats DEL TRAINING (no leakage) ─────────────
    print(f"\n  Denormalizzazione (stats dal training)...")
    gen_raw = denormalize_phase_space(gen_norm, stats)

    # ── 6. Valutazione ──────────────────────────────────────────────────────
    print(f"\n  Calcolo metriche e generazione plot...")
    output_dir = run_dir / "eval_final"
    report = evaluate_model(
        real_raw, gen_raw,
        model_name=run_name,
        output_dir=str(output_dir),
    )

    print(f"\n{'='*60}")
    print(f"  Valutazione completata: {run_name}")
    print(f"  Report: {output_dir}/{run_name}_report.json")
    print(f"{'='*60}")

    return report


if __name__ == "__main__":
    main()
