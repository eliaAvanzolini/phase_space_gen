import json
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.synthetic_linac import denormalize_phase_space
from evaluate_interpolation_10mv import MODELS, _load_cfm

N_SAMPLES = 200_000
ENERGY = 10.0
MAX_R_CM = 10.0
N_BINS = 40
OUT_PATH = Path("outputs/dose_interpolation_10mv/solver_comparison_radial.png")


def compute_radial_fluence(x_cm, y_cm, n_bins, max_r):
  r = np.sqrt(x_cm**2 + y_cm**2)
  bin_edges = np.linspace(0, max_r, n_bins + 1)
  bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
  counts, _ = np.histogram(r, bins=bin_edges)
  bin_areas = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)
  fluence = counts / bin_areas
  return bin_centers, fluence


def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Dispositivo in uso: {device}")

  print(f"Caricamento {N_SAMPLES:,} particelle dal Reference H5...")
  with h5py.File("data/energy_only_10mv_reference.h5", "r") as f:
    ps_ref = f["phase_space"][:N_SAMPLES]
  x_ref, y_ref = ps_ref[:, 0], ps_ref[:, 1]
  r_ref, flu_ref = compute_radial_fluence(x_ref, y_ref, N_BINS, MAX_R_CM)
  ref_peak = np.max(flu_ref)

  cfg = MODELS["cfm"]
  with open(cfg["stats_json"]) as f:
    stats = json.load(f)
  dim = len(stats.get("col_names", ["x", "y", "dx", "dy", "dz", "E"]))

  cond_stats_path = Path(cfg["stats_json"]).parent / "condition_stats.json"
  with open(cond_stats_path) as f:
    cond_stats = json.load(f)
  mu_c = np.array(cond_stats["mu"], dtype=np.float32)
  sig_c = np.array(cond_stats["sigma"], dtype=np.float32)
  cond_norm = (
      (np.array([ENERGY], dtype=np.float32) - mu_c) / sig_c
  ).tolist()
  cond_tensor = (
      torch.tensor(cond_norm, dtype=torch.float32, device=device)
      .unsqueeze(0)
      .repeat(N_SAMPLES, 1)
  )

  print("Caricamento modello CFM...")
  model = _load_cfm(cfg["checkpoint"], dim, device)

  solvers = [
      ("Dopri5 (usato nel run)", "dopri5", None),
      ("Euler (30 steps)", "euler", 30),
      ("Euler (100 steps)", "euler", 100),
  ]

  results = {}
  with torch.no_grad():
    for name, method, steps in solvers:
      print(f"Generazione campioni con {name}...")
      if method == "euler":
        raw = model.sample_fast(N_SAMPLES, c=cond_tensor, n_steps=steps)
      else:
        raw = model.sample(
            N_SAMPLES, cond_tensor, method="dopri5", atol=1e-4, rtol=1e-4
        )

      ps_norm = raw.cpu().numpy()
      ps_out = denormalize_phase_space(ps_norm, stats).astype(np.float32)
      x_cm, y_cm = ps_out[:, 0], ps_out[:, 1]
      r_centers, flu = compute_radial_fluence(x_cm, y_cm, N_BINS, MAX_R_CM)
      results[name] = flu

  fig, (ax1, ax2) = plt.subplots(
      2, 1, figsize=(8, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
  )

  ax1.plot(
      r_ref,
      flu_ref / ref_peak * 100,
      label="Reference MC",
      color="black",
      lw=2.0,
  )
  colors = ["tab:blue", "tab:orange", "tab:green"]
  for (name, _, _), col in zip(solvers, colors):
    ax1.plot(
        r_ref,
        results[name] / ref_peak * 100,
        label=name,
        color=col,
        lw=1.5,
        ls="--",
    )

  ax1.set_ylabel("Fluenza relativa radiale (%)")
  ax1.set_title("Confronto fluenza radiale del phase space: Solutori ODE vs MC")
  ax1.axvspan(6.0, 8.5, color="red", alpha=0.1, label="Zona penombra (anello)")
  ax1.grid(alpha=0.3)
  ax1.legend()

  for (name, _, _), col in zip(solvers, colors):
    ratio = (results[name] - flu_ref) / (flu_ref + 1e-8) * 100
    ax2.plot(r_ref, ratio, label=name, color=col, lw=1.5)

  ax2.axhline(0, color="black", lw=1.0)
  ax2.axhline(-10, color="grey", ls=":", lw=1)
  ax2.axhline(10, color="grey", ls=":", lw=1)
  ax2.axvspan(6.0, 8.5, color="red", alpha=0.1)
  ax2.set_xlabel("Raggio dall'asse centrale (cm)")
  ax2.set_ylabel("Δ Fluenza / Ref (%)")
  ax2.set_ylim(-60, 40)
  ax2.grid(alpha=0.3)

  plt.tight_layout()
  OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(OUT_PATH, dpi=130)
  plt.close()
  print(f"\n✅ Grafico salvato in: {OUT_PATH}")


if __name__ == "__main__":
  main()
