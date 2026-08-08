import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

MODELS = {
    "cfm": {
        "checkpoint": "outputs/cfm_energy_only/best_model.pt",
        "stats_json": "outputs/cfm_energy_only/normalization_stats.json",
        "model_type": "cfm",
    },
    "nsf": {
        "checkpoint": "outputs/nsf_energy_only/best_model.pt",
        "stats_json": "outputs/nsf_energy_only/normalization_stats.json",
        "model_type": "nsf",
    },
    "gan": {
        "checkpoint": "outputs/gan_energy_only/best_model.pt",
        "stats_json": "outputs/gan_energy_only/normalization_stats.json",
        "model_type": "gan",
    },
}

REFERENCE_10MV_PATH = "data/energy_only_10mv_reference.h5"


def _load_cfm(checkpoint_path, dim, device):
    from models.cfm import CFMTrainer, PhaseSpaceCFM
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model") or ckpt
    hidden_dim = 256
    if "velocity_net.input_proj.weight" in sd:
        hidden_dim = sd["velocity_net.input_proj.weight"].shape[0]
    n_layers = 4
    max_idx = -1
    for k in sd:
        if "velocity_net.res_layers." in k:
            max_idx = max(max_idx, int(k.split("res_layers.")[1].split(".")[0]))
    if max_idx >= 0:
        n_layers = max_idx + 1
    model = PhaseSpaceCFM(dim=dim, cond_dim=1, hidden_dim=hidden_dim, n_layers=n_layers)
    trainer = CFMTrainer(model, device=device, lr=1e-4)
    trainer.load(checkpoint_path)
    return model.to(device).eval()


def _load_nsf(checkpoint_path, dim, device):
    from models.nsf import NSFTrainer, PhaseSpaceNSF
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model") or ckpt

    # Legge gli iperparametri VERI usati in training dal config.json del run,
    # invece di indovinarli (n_bins hardcodato a 16 non combaciava col default
    # reale di train.py, 8, causando un mismatch di shape nel caricamento).
    config_path = Path(checkpoint_path).parent / "config.json"
    n_bins = 8
    n_transforms = 6
    tail_bound = 5.0
    if config_path.exists():
        with open(config_path) as f:
            run_config = json.load(f)
        n_bins = run_config.get("n_bins", n_bins)
        n_transforms = run_config.get("n_transforms", n_transforms)
        tail_bound = run_config.get("tail_bound", tail_bound)
    else:
        print(f"  [WARNING] {config_path} non trovato, uso n_bins/n_transforms/tail_bound "
              f"di default - potrebbe non combaciare con l'architettura salvata nel checkpoint.")

    hidden_dim = 256
    for k, v in sd.items():
        if "transform_net.initial_layer.weight" in k:
            hidden_dim = v.shape[0]
            break

    model = PhaseSpaceNSF(dim=dim, cond_dim=1, n_transforms=n_transforms, hidden_dim=hidden_dim,
                            n_bins=n_bins, tail_bound=tail_bound)
    trainer = NSFTrainer(model, device=device, lr=1e-4)
    trainer.load(checkpoint_path)
    return model.to(device).eval()


def _load_gan_auto(checkpoint_path, cond_dim, out_dim, device):
    from models.gan import PhaseSpaceGenerator
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("generator") or ckpt.get("G") or ckpt
    clean_state = {k.replace("model.", ""): v for k, v in state_dict.items()}
    hidden_dims = []
    idx = 0
    while f"trunk.{idx}.0.weight" in clean_state:
        hidden_dims.append(clean_state[f"trunk.{idx}.0.weight"].shape[0])
        idx += 1
    if not hidden_dims:
        hidden_dims = [256, 512, 512, 256]
    G = PhaseSpaceGenerator(latent_dim=64, cond_dim=cond_dim, hidden_dims=hidden_dims, output_dim=out_dim).to(device)
    G.load_state_dict(clean_state)
    return G.eval()


def generate_at_energy(model_name, energy, n_samples, device, chunk_size=250_000):
    """Genera phase space alla condizione data (E=10.0, mai vista in training),
    con retry robusto su eventuali instabilita' numeriche NSF."""
    from data.synthetic_linac import denormalize_phase_space

    cfg = MODELS[model_name]
    with open(cfg["stats_json"]) as f:
        stats = json.load(f)
    dim = len(stats.get("col_names", ["x", "y", "dx", "dy", "dz", "E"]))

    cond_stats_path = Path(cfg["stats_json"]).parent / "condition_stats.json"
    with open(cond_stats_path) as f:
        cond_stats = json.load(f)
    mu_c = np.array(cond_stats["mu"], dtype=np.float32)
    sig_c = np.array(cond_stats["sigma"], dtype=np.float32)
    cond_norm = ((np.array([energy], dtype=np.float32) - mu_c) / sig_c).tolist()

    if cfg["model_type"] == "gan":
        ckpt = torch.load(cfg["checkpoint"], map_location=device, weights_only=False)
        state_dict = ckpt.get("generator") or ckpt.get("G") or ckpt
        clean_state = {k.replace("model.", ""): v for k, v in state_dict.items()}
        out_dim = clean_state["head.weight"].shape[0] if "head.weight" in clean_state else dim
        model = _load_gan_auto(cfg["checkpoint"], cond_dim=1, out_dim=out_dim, device=device)
    elif cfg["model_type"] == "cfm":
        model = _load_cfm(cfg["checkpoint"], dim, device)
    else:
        model = _load_nsf(cfg["checkpoint"], dim, device)

    chunks = []
    n_done = 0
    with torch.no_grad():
        while n_done < n_samples:
            n_chunk = min(chunk_size, n_samples - n_done)
            cond_tensor = torch.tensor(cond_norm, dtype=torch.float32, device=device).unsqueeze(0).repeat(n_chunk, 1)

            chunk_out = None
            for attempt in range(3):
                try:
                    if cfg["model_type"] == "gan":
                        z = torch.randn(n_chunk, 64, device=device)
                        chunk_out = model(z, cond_tensor).cpu().numpy()
                        del z
                    elif cfg["model_type"] == "cfm":
                        chunk_out = model.sample(n_chunk, cond_tensor, method="dopri5", atol=1e-4, rtol=1e-4).cpu().numpy()
                    else:
                        chunk_out = model.sample(n_chunk, cond_tensor).cpu().numpy()
                    break
                except AssertionError:
                    print(f"  [WARNING] instabilita' numerica ({model_name}, chunk {n_done}), tentativo {attempt+1}/3...")
                    torch.manual_seed(1000 + n_done + attempt)

            if chunk_out is None:
                print(f"  [WARNING] chunk da {n_done:,} scartato dopo 3 tentativi falliti")
                n_done += n_chunk
                del cond_tensor
                continue

            chunks.append(chunk_out)
            n_done += n_chunk
            del cond_tensor
            if device == "cuda":
                torch.cuda.empty_cache()

    ps_norm = np.concatenate(chunks, axis=0)
    ps_out = denormalize_phase_space(ps_norm, stats).astype(np.float32)

    if ps_out.shape[1] == 6:
        ps_7d = np.zeros((len(ps_out), 7), dtype=np.float32)
        ps_7d[:, [0, 1, 3, 4, 5, 6]] = ps_out
        ps_7d[:, 2] = stats.get("z_const", 0.0)
        ps_out = ps_7d

    return ps_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", type=float, default=10.0)
    ap.add_argument("--n_samples", type=int, default=500_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_dir", default="outputs/interpolation_10mv_eval")
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                     choices=list(MODELS.keys()),
                     help="Quali modelli valutare (default: tutti). Utile per "
                          "valutare solo i modelli gia' finiti di addestrare, "
                          "senza caricare checkpoint parziali di quelli ancora "
                          "in training.")
    args = ap.parse_args()

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f" VALUTAZIONE INTERPOLAZIONE A {args.energy} MeV (energia nascosta, mai in training)")
    print("=" * 70)

    print(f"\nCaricamento reference reale ({REFERENCE_10MV_PATH})...")
    with h5py.File(REFERENCE_10MV_PATH, "r") as f:
        n_available = f["phase_space"].shape[0]
        n_ref = min(n_available, args.n_samples)
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n_available, size=n_ref, replace=False))
        ps_real = f["phase_space"][idx]
    print(f"  {len(ps_real):,} particelle reali campionate (su {n_available:,} disponibili)")

    from evaluate import evaluate_model

    all_reports = {}
    for model_name in args.models:
        print(f"\n{'-'*70}\nModello: {model_name.upper()}\n{'-'*70}")
        ps_gen = generate_at_energy(model_name, args.energy, len(ps_real), device)
        print(f"  {len(ps_gen):,} particelle generate a {args.energy}MeV")

        report = evaluate_model(
            ps_real, ps_gen,
            model_name=f"{model_name}_interp10mv",
            output_dir=str(out / model_name),
        )
        all_reports[model_name] = report

    summary_path = out / "interpolation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"\n{'='*70}\nRiepilogo completo salvato in: {summary_path}\n{'='*70}")


if __name__ == "__main__":
    main()
