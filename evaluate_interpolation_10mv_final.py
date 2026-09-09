import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import uproot

MODELS = {
    "cfm": {
        "checkpoint": "outputs/cfm_energy_only/best_model.pt",
        "stats_json": "outputs/cfm_energy_only/normalization_stats.json",
        "model_type": "cfm",
        "chunk_size": 250_000,
    },
    "nsf": {
        "checkpoint": "outputs/nsf_energy_only/best_model.pt",
        "stats_json": "outputs/nsf_energy_only/normalization_stats.json",
        "model_type": "nsf",
        "chunk_size": 50_000,  # Chunk ridotto per evitare OOM su spline inversion
    },
    "gan": {
        "checkpoint": "outputs/gan_energy_only/best_model.pt",
        "stats_json": "outputs/gan_energy_only/normalization_stats.json",
        "model_type": "gan",
        "chunk_size": 250_000,
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
  model = PhaseSpaceCFM(
      dim=dim, cond_dim=1, hidden_dim=hidden_dim, n_layers=n_layers
  )
  trainer = CFMTrainer(model, device=device, lr=1e-4)
  trainer.load(checkpoint_path)
  return model.to(device).eval()


def _load_nsf(checkpoint_path, dim, device):
  from models.nsf import NSFTrainer, PhaseSpaceNSF

  ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
  sd = ckpt.get("model") or ckpt

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

  hidden_dim = 256
  for k, v in sd.items():
    if "transform_net.initial_layer.weight" in k:
      hidden_dim = v.shape[0]
      break

  model = PhaseSpaceNSF(
      dim=dim,
      cond_dim=1,
      n_transforms=n_transforms,
      hidden_dim=hidden_dim,
      n_bins=n_bins,
      tail_bound=tail_bound,
  )
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
  G = PhaseSpaceGenerator(
      latent_dim=64, cond_dim=cond_dim, hidden_dims=hidden_dims, output_dim=out_dim
  ).to(device)
  G.load_state_dict(clean_state)
  return G.eval()


def apply_physical_filters(ps: np.ndarray) -> np.ndarray:
  """Filtra energie <= 0.01 MeV e rinormalizza i coseni direttori."""
  n_init = len(ps)

  # 1. Filtro energetico (taglio fotoni fantasma / negativi)
  valid_mask = ps[:, 6] >= 0.01
  n_invalid_e = n_init - int(valid_mask.sum())
  if n_invalid_e > 0:
    pct = (n_invalid_e / n_init) * 100
    print(
        f"  [Filtro Fisico] Scartate {n_invalid_e:,} particelle con E < 0.01"
        f" MeV ({pct:.3f}%)"
    )
  ps = ps[valid_mask]

  # 2. Rinormalizzazione versore di direzione (u, v, w)
  uvw = ps[:, 3:6]
  norm = np.linalg.norm(uvw, axis=1, keepdims=True)
  norm[norm == 0] = 1.0
  ps[:, 3:6] = uvw / norm

  # 3. Direzione z sempre verso il fantoccio (dz > 0)
  ps[:, 5] = np.abs(ps[:, 5])

  return ps


def write_phsp_root(ps: np.ndarray, out_path: Path):
  """Scrive il phase space in formato ROOT per OpenGATE (mm, MeV)."""
  tree_dict = {
      "X": (ps[:, 0] * 10.0).astype(np.float32),  # cm -> mm
      "Y": (ps[:, 1] * 10.0).astype(np.float32),
      "Z": np.zeros(len(ps), dtype=np.float32),  # Sorgente a Z=0
      "dX": ps[:, 3].astype(np.float32),
      "dY": ps[:, 4].astype(np.float32),
      "dZ": ps[:, 5].astype(np.float32),
      "E": ps[:, 6].astype(np.float32),
      "Weight": np.ones(len(ps), dtype=np.float32),
  }
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with uproot.recreate(out_path) as f:
    f["PhaseSpace"] = tree_dict
  print(
      f"  [ROOT] Scritto phase space per GATE: {out_path} ({len(ps):,} particelle)"
  )


def generate_at_energy(
    model_name: str, energy: float, n_samples: int, device: str
) -> np.ndarray:
  """Genera phase space con retry su instabilità numeriche e filtri fisici."""
  from data.synthetic_linac import denormalize_phase_space

  cfg = MODELS[model_name]
  chunk_size = cfg.get("chunk_size", 100_000)

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
    out_dim = (
        clean_state["head.weight"].shape[0]
        if "head.weight" in clean_state
        else dim
    )
    model = _load_gan_auto(
        cfg["checkpoint"], cond_dim=1, out_dim=out_dim, device=device
    )
  elif cfg["model_type"] == "cfm":
    model = _load_cfm(cfg["checkpoint"], dim, device)
  else:
    model = _load_nsf(cfg["checkpoint"], dim, device)

  chunks = []
  n_done = 0
  with torch.no_grad():
    while n_done < n_samples:
      n_chunk = min(chunk_size, n_samples - n_done)
      cond_tensor = (
          torch.tensor(cond_norm, dtype=torch.float32, device=device)
          .unsqueeze(0)
          .repeat(n_chunk, 1)
      )

      chunk_out = None
      for attempt in range(3):
        try:
          if cfg["model_type"] == "gan":
            z = torch.randn(n_chunk, 64, device=device)
            chunk_out = model(z, cond_tensor).cpu().numpy()
            del z
          elif cfg["model_type"] == "cfm":
            chunk_out = model.sample_fast(
                n_chunk, c=cond_tensor, n_steps=30
            ).cpu().numpy()
          else:
            chunk_out = model.sample(n_chunk, cond_tensor).cpu().numpy()
          break
        except (AssertionError, RuntimeError) as e:
          print(
              f"  [ATTEMPT] Instabilità ({model_name}, chunk {n_done}): {e}."
              f" Tentativo {attempt+1}/3..."
          )
          torch.manual_seed(1000 + n_done + attempt)

      if chunk_out is None:
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

  # Applicazione filtri fisici
  ps_clean = apply_physical_filters(ps_out)
  return ps_clean


def compute_radial_fluence(x_cm, y_cm, n_bins=40, max_r=10.0):
  r = np.sqrt(x_cm**2 + y_cm**2)
  bin_edges = np.linspace(0, max_r, n_bins + 1)
  bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
  counts, _ = np.histogram(r, bins=bin_edges)
  bin_areas = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)
  fluence = counts / bin_areas
  return bin_centers, fluence


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--energy", type=float, default=10.0)
  ap.add_argument("--n_samples", type=int, default=1_000_000)
  ap.add_argument("--device", default="cuda")
  ap.add_argument("--output_dir", default="outputs/interpolation_10mv_eval")
  ap.add_argument(
      "--models",
      nargs="+",
      default=["cfm", "nsf", "gan"],
      choices=["cfm", "nsf", "gan"],
  )
  ap.add_argument(
      "--save_root",
      action="store_true",
      help="Salva i file .root per OpenGATE (gen_<model>.root)",
  )
  args = ap.parse_args()

  device = (
      "cuda"
      if args.device == "cuda" and torch.cuda.is_available()
      else "cpu"
  )
  out = Path(args.output_dir)
  out.mkdir(parents=True, exist_ok=True)

  print("=" * 70)
  print(f" VALUTAZIONE MULTI-MODELLO A {args.energy} MeV (DEVICE: {device})")
  print("=" * 70)

  print(f"\nCaricamento Reference reale ({REFERENCE_10MV_PATH})...")
  with h5py.File(REFERENCE_10MV_PATH, "r") as f:
    n_available = f["phase_space"].shape[0]
    n_ref = min(n_available, args.n_samples)
    ps_real = f["phase_space"][:n_ref]

  r_centers, flu_ref = compute_radial_fluence(ps_real[:, 0], ps_real[:, 1])
  ref_peak = np.max(flu_ref)

  fig, (ax1, ax2) = plt.subplots(
      2, 1, figsize=(8, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
  )
  ax1.plot(
      r_centers,
      flu_ref / ref_peak * 100,
      label="Reference MC",
      color="black",
      lw=2.0,
  )

  colors = {"cfm": "tab:blue", "nsf": "tab:purple", "gan": "tab:orange"}

  for model_name in args.models:
    print(f"\n{'-'*70}\nGenerazione: {model_name.upper()}\n{'-'*70}")
    ps_gen = generate_at_energy(model_name, args.energy, n_ref, device)

    if args.save_root:
      root_path = out / f"gen_{model_name}.root"
      write_phsp_root(ps_gen, root_path)

    # Fluenza radiale
    _, flu_mod = compute_radial_fluence(ps_gen[:, 0], ps_gen[:, 1])
    ax1.plot(
        r_centers,
        flu_mod / ref_peak * 100,
        label=model_name.upper(),
        color=colors[model_name],
        lw=1.6,
        ls="--",
    )

    ratio = (flu_mod - flu_ref) / (flu_ref + 1e-8) * 100
    ax2.plot(
        r_centers,
        ratio,
        label=model_name.upper(),
        color=colors[model_name],
        lw=1.6,
    )

  ax1.set_ylabel("Fluenza relativa radiale (%)")
  ax1.set_title(
      f"Confronto Fluenza Radiale a {args.energy} MV: CFM vs NSF vs GAN"
  )
  ax1.axvspan(6.0, 8.5, color="red", alpha=0.1, label="Zona Penombra (7 cm)")
  ax1.grid(alpha=0.3)
  ax1.legend()

  ax2.axhline(0, color="black", lw=1.0)
  ax2.axhline(-10, color="grey", ls=":", lw=1)
  ax2.axhline(10, color="grey", ls=":", lw=1)
  ax2.axvspan(6.0, 8.5, color="red", alpha=0.1)
  ax2.set_xlabel("Raggio dall'asse centrale (cm)")
  ax2.set_ylabel("Δ Fluenza / Ref (%)")
  ax2.set_ylim(-60, 40)
  ax2.grid(alpha=0.3)

  plot_path = out / "radial_comparison_all_models.png"
  plt.tight_layout()
  plt.savefig(plot_path, dpi=130)
  plt.close()
  print(f"\n{'='*70}\n✅ Grafico comparativo salvato in: {plot_path}\n{'='*70}")


if __name__ == "__main__":
  main()
