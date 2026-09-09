"""
plot_reports_dose_extended.py
==============================
Report dosimetrico esteso per il confronto reference (GATE MC) vs modelli
generativi condizionati (CFM/NSF).

Include:
  1. Mappa di significatività statistica z = (D_mod - D_ref) / sigma_combinata.
  2. Analisi quantitativa del bordo campo (edge 50%, penombra 80%-20%).
  3. Gamma Index 3D (3%/3mm e 2%/2mm clinico).
  4. Salvataggio incrementale progressivo (evita perdite di dati su timeout).
  5. Supporto CLI per selezionare campi specifici o saltare quelli già calcolati.
"""

import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymedphys
import SimpleITK as sitk

BASE_DIR = "outputs/dose_validation"
FIELDS = [
    "6mv_5x5",
    "6mv_10x10",
    "6mv_20x20",
    "10mv_5x5",
    "10mv_10x10",
    "10mv_20x20",
]
MODELS = ["cfm", "nsf"]

CRITERIA = [
    {"name": "3pct_3mm", "dose_pct": 3.0, "dist_mm": 3.0},
    {"name": "2pct_2mm_clinico", "dose_pct": 2.0, "dist_mm": 2.0},
]
LOWER_DOSE_CUTOFF = 10.0
HALF_WIDTH_VOX = 3

# Config incertezza statistica
ANALYZE_UNCERTAINTY = True
UNCERTAINTY_MODE = "auto"
Z_ABS_MAX_PLOT = 5.0

# Config analisi bordo campo / penombra
ANALYZE_FIELD_EDGES = True
EDGE_SEARCH_WINDOW_CM = 2.0


def load_dose(path):
  img = sitk.ReadImage(str(path))
  arr = sitk.GetArrayFromImage(img).astype(np.float64)  # (Z, Y, X)
  spacing_zyx = img.GetSpacing()[::-1]
  return arr, spacing_zyx


def robust_max(arr):
  nz = arr[arr > 0]
  return float(np.percentile(nz, 99.9)) if len(nz) > 0 else float(arr.max())


def extract_pdd(arr, cy, cx, hw):
  y0, y1 = max(0, cy - hw), min(arr.shape[1], cy + hw + 1)
  x0, x1 = max(0, cx - hw), min(arr.shape[2], cx + hw + 1)
  return arr[:, y0:y1, x0:x1].mean(axis=(1, 2))


def extract_transverse(arr, z_idx, cy, hw):
  y0, y1 = max(0, cy - hw), min(arr.shape[1], cy + hw + 1)
  return arr[z_idx, y0:y1, :].mean(axis=0)


def find_uncertainty_path(dose_path: Path) -> Path | None:
  stem = dose_path.name
  candidates = [
      dose_path.with_name(stem.replace("_dose.mhd", "_dose_uncertainty.mhd")),
      dose_path.with_name(stem.replace(".mhd", "_uncertainty.mhd")),
      dose_path.with_name(stem.replace("_dose.mhd", "-dose_uncertainty.mhd")),
  ]
  for c in candidates:
    if c.exists():
      return c
  run_tag = stem.split("_dose")[0]
  for p in dose_path.parent.glob(f"*{run_tag}*uncertain*.mhd"):
    return p
  return None


def _to_absolute_uncertainty(
    unc_arr: np.ndarray, dose_arr: np.ndarray, mode: str
) -> np.ndarray:
  if mode == "relative":
    return unc_arr * dose_arr
  if mode == "absolute":
    return unc_arr.copy()

  nz = unc_arr[unc_arr > 0]
  if len(nz) == 0:
    return unc_arr.copy()
  p90 = np.percentile(nz, 90)
  inferred = "relative" if p90 < 2.0 else "absolute"
  print(
      f"    [auto] incertezza rilevata come '{inferred}' (p90 valori grezzi ="
      f" {p90:.3g})"
  )
  return unc_arr * dose_arr if inferred == "relative" else unc_arr.copy()


def load_uncertainty_absolute(dose_path: Path, dose_arr: np.ndarray, mode: str):
  unc_path = find_uncertainty_path(dose_path)
  if unc_path is None:
    return None
  unc_img, unc_spacing = load_dose(unc_path)
  return _to_absolute_uncertainty(unc_img, dose_arr, mode)


def parse_field_halfwidth_cm(field_name: str):
  m = re.search(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)", field_name)
  if not m:
    return None
  w = float(m.group(1))
  return w / 2.0


def _local_field_norm(
    x_cm: np.ndarray, y: np.ndarray, half_width_cm: float
) -> np.ndarray:
  inner = np.abs(x_cm) < (half_width_cm * 0.6)
  ref100 = np.mean(y[inner]) if inner.sum() > 0 else np.max(y)
  if ref100 <= 0:
    return np.full_like(y, np.nan)
  return y / ref100 * 100.0


def _find_level_crossing(x_cm, y_local, level, edge_nominal_cm, window_cm):
  lo, hi = edge_nominal_cm - window_cm, edge_nominal_cm + window_cm
  m = (x_cm >= min(lo, hi)) & (x_cm <= max(lo, hi))
  xs, ys = x_cm[m], y_local[m]
  if len(xs) < 2:
    return float("nan")
  order = np.argsort(xs)
  xs, ys = xs[order], ys[order]
  diffs = ys - level
  sign_changes = np.where(np.diff(np.sign(diffs)) != 0)[0]
  if len(sign_changes) == 0:
    return float("nan")
  i = sign_changes[0]
  x1, x2, y1, y2 = xs[i], xs[i + 1], ys[i], ys[i + 1]
  if y2 == y1:
    return float(x1)
  return float(x1 + (level - y1) * (x2 - x1) / (y2 - y1))


def analyze_field_edge(
    x_cm, y_ref, y_model, half_width_cm, window_cm=EDGE_SEARCH_WINDOW_CM
):
  y_ref_local = _local_field_norm(x_cm, y_ref, half_width_cm)
  y_model_local = _local_field_norm(x_cm, y_model, half_width_cm)

  out = {}
  for side, sign in [("left", -1.0), ("right", +1.0)]:
    edge_nom = sign * half_width_cm
    res = {}
    for tag, y_local in [("ref", y_ref_local), ("model", y_model_local)]:
      e50 = _find_level_crossing(x_cm, y_local, 50.0, edge_nom, window_cm)
      e80 = _find_level_crossing(x_cm, y_local, 80.0, edge_nom, window_cm)
      e20 = _find_level_crossing(x_cm, y_local, 20.0, edge_nom, window_cm)
      penumbra_mm = (
          abs(e20 - e80) * 10.0
          if np.isfinite(e20) and np.isfinite(e80)
          else float("nan")
      )
      res[tag] = {"edge50_cm": e50, "penumbra_80_20_mm": penumbra_mm}
    shift_mm = (
        (res["model"]["edge50_cm"] - res["ref"]["edge50_cm"]) * 10.0
        if np.isfinite(res["model"]["edge50_cm"])
        and np.isfinite(res["ref"]["edge50_cm"])
        else float("nan")
    )
    res["edge_shift_model_vs_ref_mm"] = shift_mm
    out[side] = res
  return out, y_ref_local, y_model_local


def process_pair(
    field_dir,
    field,
    model,
    ref_arr,
    ref_norm,
    ref_gnorm,
    spacing_zyx,
    out_dir,
    ref_dose_path,
):
  sp_z, sp_y, sp_x = spacing_zyx
  nz, ny, nx = ref_arr.shape
  cz, cy, cx = nz // 2, ny // 2, nx // 2

  model_path = field_dir / f"dose_{model}_dose.mhd"
  if not model_path.exists():
    print(f"  ⚠️ [{model.upper()}] mappa non trovata in {model_path}, salto.")
    return None

  model_arr, model_spacing = load_dose(model_path)
  if not np.allclose(model_spacing, spacing_zyx, rtol=1e-3):
    print(
        f"  ⚠️ spacing diverso tra reference ({spacing_zyx}) e"
        f" {model.upper()} ({model_spacing})"
    )
  assert ref_arr.shape == model_arr.shape, (
      f"Shape mismatch {field}/{model}: ref={ref_arr.shape} vs"
      f" model={model_arr.shape} (controlla che voxel_mm sia stato lo stesso in"
      " entrambe le run)"
  )

  model_gnorm = robust_max(model_arr)
  model_norm = (model_arr / model_gnorm) * 100.0

  # 1. Mappe 2D
  depth_cm_slice = 0.0
  z_idx_slice = max(0, min(int(round(depth_cm_slice * 10.0 / sp_z)), nz - 1))
  extent_xy = [
      -nx / 2 * sp_x / 10,
      nx / 2 * sp_x / 10,
      -ny / 2 * sp_y / 10,
      ny / 2 * sp_y / 10,
  ]
  extent_xz = [-nx / 2 * sp_x / 10, nx / 2 * sp_x / 10, 0, nz * sp_z / 10]
  vmax = max(ref_norm.max(), model_norm.max())

  fig, axes = plt.subplots(2, 3, figsize=(18, 11))
  im0 = axes[0, 0].imshow(
      ref_norm[z_idx_slice],
      extent=extent_xy,
      origin="lower",
      cmap="jet",
      vmin=0,
      vmax=vmax,
  )
  axes[0, 0].set_title(f"REFERENCE — XY @ Z={depth_cm_slice}cm ({field})")
  axes[0, 0].set_xlabel("X (cm)")
  axes[0, 0].set_ylabel("Y (cm)")
  plt.colorbar(im0, ax=axes[0, 0], label="Dose (% P99.9)")

  im1 = axes[0, 1].imshow(
      model_norm[z_idx_slice],
      extent=extent_xy,
      origin="lower",
      cmap="jet",
      vmin=0,
      vmax=vmax,
  )
  axes[0, 1].set_title(f"{model.upper()} — XY @ Z={depth_cm_slice}cm")
  axes[0, 1].set_xlabel("X (cm)")
  axes[0, 1].set_ylabel("Y (cm)")
  plt.colorbar(im1, ax=axes[0, 1], label="Dose (% P99.9)")

  diff_xy = model_norm[z_idx_slice] - ref_norm[z_idx_slice]
  dmax_abs = np.percentile(np.abs(diff_xy), 99) or 1.0
  im2 = axes[0, 2].imshow(
      diff_xy,
      extent=extent_xy,
      origin="lower",
      cmap="RdBu_r",
      vmin=-dmax_abs,
      vmax=dmax_abs,
  )
  axes[0, 2].set_title(f"Differenza {model.upper()} - Reference (XY)")
  axes[0, 2].set_xlabel("X (cm)")
  axes[0, 2].set_ylabel("Y (cm)")
  plt.colorbar(im2, ax=axes[0, 2], label="Δ Dose (% P99.9)")

  im3 = axes[1, 0].imshow(
      ref_norm[:, cy, :],
      extent=extent_xz,
      origin="lower",
      cmap="jet",
      vmin=0,
      vmax=vmax,
      aspect="auto",
  )
  axes[1, 0].set_title("REFERENCE — XZ (assiale)")
  axes[1, 0].set_xlabel("X (cm)")
  axes[1, 0].set_ylabel("Profondita' Z (cm)")
  plt.colorbar(im3, ax=axes[1, 0], label="Dose (% P99.9)")

  im4 = axes[1, 1].imshow(
      model_norm[:, cy, :],
      extent=extent_xz,
      origin="lower",
      cmap="jet",
      vmin=0,
      vmax=vmax,
      aspect="auto",
  )
  axes[1, 1].set_title(f"{model.upper()} — XZ (assiale)")
  axes[1, 1].set_xlabel("X (cm)")
  axes[1, 1].set_ylabel("Profondita' Z (cm)")
  plt.colorbar(im4, ax=axes[1, 1], label="Dose (% P99.9)")

  diff_xz = model_norm[:, cy, :] - ref_norm[:, cy, :]
  dmax_abs_xz = np.percentile(np.abs(diff_xz), 99) or 1.0
  im5 = axes[1, 2].imshow(
      diff_xz,
      extent=extent_xz,
      origin="lower",
      cmap="RdBu_r",
      vmin=-dmax_abs_xz,
      vmax=dmax_abs_xz,
      aspect="auto",
  )
  axes[1, 2].set_title(f"Differenza {model.upper()} - Reference (XZ)")
  axes[1, 2].set_xlabel("X (cm)")
  axes[1, 2].set_ylabel("Profondita' Z (cm)")
  plt.colorbar(im5, ax=axes[1, 2], label="Δ Dose (% P99.9)")

  plt.tight_layout()
  maps_path = out_dir / f"dose_maps_2d_{model}.png"
  plt.savefig(maps_path, dpi=130)
  plt.close(fig)

  # 2. Profili 1D
  half_width_cm = (
      parse_field_halfwidth_cm(field) if ANALYZE_FIELD_EDGES else None
  )
  n_panels = 4 if half_width_cm is not None else 3
  fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
  z_axis_cm = np.arange(nz) * sp_z / 10.0
  x_axis_cm = (np.arange(nx) - cx) * sp_x / 10.0

  ref_pdd = extract_pdd(ref_norm, cy, cx, HALF_WIDTH_VOX)
  model_pdd = extract_pdd(model_norm, cy, cx, HALF_WIDTH_VOX)
  axes[0].plot(z_axis_cm, ref_pdd, label="REFERENCE", color="black")
  axes[0].plot(z_axis_cm, model_pdd, label=model.upper(), color="tab:blue")
  axes[0].set_xlabel("Profondita' Z (cm)")
  axes[0].set_ylabel("Dose (% P99.9)")
  axes[0].set_title(f"PDD ({field}, mediato 7x7 voxel)")
  axes[0].legend()
  axes[0].grid(alpha=0.3)

  edge_results_by_depth = {}
  for i, depth_cm in enumerate([3.0, 10.0]):
    z_idx = max(0, min(int(round(depth_cm * 10.0 / sp_z)), nz - 1))
    ref_t = extract_transverse(ref_norm, z_idx, cy, HALF_WIDTH_VOX)
    model_t = extract_transverse(model_norm, z_idx, cy, HALF_WIDTH_VOX)
    axes[i + 1].plot(x_axis_cm, ref_t, label="REFERENCE", color="black")
    axes[i + 1].plot(x_axis_cm, model_t, label=model.upper(), color="tab:blue")
    axes[i + 1].set_xlabel("Posizione X (cm)")
    axes[i + 1].set_ylabel("Dose (% P99.9)")
    axes[i + 1].set_title(f"Trasversale @ Z={depth_cm}cm")
    axes[i + 1].legend()
    axes[i + 1].grid(alpha=0.3)

    if half_width_cm is not None:
      edge_res, ref_local, model_local = analyze_field_edge(
          x_axis_cm, ref_t, model_t, half_width_cm
      )
      edge_results_by_depth[depth_cm] = edge_res
      if depth_cm == 10.0:
        ax = axes[n_panels - 1]
        m = np.abs(x_axis_cm - half_width_cm) < (EDGE_SEARCH_WINDOW_CM + 0.5)
        ax.plot(x_axis_cm[m], ref_local[m], label="REFERENCE", color="black")
        ax.plot(
            x_axis_cm[m], model_local[m], label=model.upper(), color="tab:blue"
        )
        ax.axvline(
            half_width_cm, color="grey", ls="--", lw=1, label="Jaw nominale"
        )
        ax.axhline(50, color="green", ls=":", lw=1)
        ax.set_xlabel("Posizione X (cm)")
        ax.set_ylabel("Dose (% plateau locale)")
        ax.set_title(f"Zoom bordo campo (+X) @ Z=10cm\n{field}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

  plt.tight_layout()
  profiles_path = out_dir / f"dose_profiles_1d_{model}.png"
  plt.savefig(profiles_path, dpi=130)
  plt.close(fig)

  # 3. Gamma index 3D + mappa 2D
  axes_3d = tuple(np.arange(s) * sp for s, sp in zip(ref_arr.shape, spacing_zyx))
  results = {}
  for crit in CRITERIA:
    gamma_map = pymedphys.gamma(
        axes_3d,
        ref_arr,
        axes_3d,
        model_arr,
        dose_percent_threshold=crit["dose_pct"],
        distance_mm_threshold=crit["dist_mm"],
        lower_percent_dose_cutoff=LOWER_DOSE_CUTOFF,
        global_normalisation=ref_gnorm,
        max_gamma=2,
        skip_once_passed=True,
        quiet=True,
    )
    valid = gamma_map[~np.isnan(gamma_map)]
    pass_rate = (
        float((valid <= 1.0).mean() * 100) if len(valid) > 0 else float("nan")
    )
    results[crit["name"]] = {"pass_rate_pct": pass_rate, "n_valid": len(valid)}

    gamma_slice = gamma_map[z_idx_slice]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        gamma_slice,
        extent=extent_xy,
        origin="lower",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=2,
    )
    ax.set_title(
        f"Gamma 2D {field} — {model.upper()} — {crit['name']}\nPass rate 3D:"
        f" {pass_rate:.1f}%"
    )
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    plt.colorbar(im, ax=ax, label="Gamma (<=1 = pass)")
    plt.tight_layout()
    plt.savefig(out_dir / f"gamma_map_2d_{model}_{crit['name']}.png", dpi=130)
    plt.close(fig)

  # 4. Mappa di significatività statistica (z-score)
  zscore_summary = None
  if ANALYZE_UNCERTAINTY:
    sigma_ref_abs = load_uncertainty_absolute(
        ref_dose_path, ref_arr, UNCERTAINTY_MODE
    )
    sigma_model_abs = load_uncertainty_absolute(
        model_path, model_arr, UNCERTAINTY_MODE
    )

    if sigma_ref_abs is None or sigma_model_abs is None:
      print(
          f"  ⚠️ [{model.upper()}] mappa di incertezza non trovata, salto"
          " analisi z-score."
      )
    else:
      # Mappa 2D sulla fetta
      mask_slice = ref_norm[z_idx_slice] >= LOWER_DOSE_CUTOFF
      denom_slice = np.sqrt(sigma_ref_abs[z_idx_slice]**2 + sigma_model_abs[z_idx_slice]**2)
      with np.errstate(divide="ignore", invalid="ignore"):
        z_map = np.where(denom_slice > 0, (model_arr[z_idx_slice] - ref_arr[z_idx_slice]) / denom_slice, np.nan)
      z_map = np.where(mask_slice, z_map, np.nan)

      # Z-score volumetrico 3D (come nel report della tesi)
      mask_3d = ref_norm >= 2.0
      denom_3d = np.sqrt(sigma_ref_abs**2 + sigma_model_abs**2)
      with np.errstate(divide="ignore", invalid="ignore"):
        z_3d = np.where((denom_3d > 0) & mask_3d, (model_arr - ref_arr) / denom_3d, np.nan)
      z_valid = z_3d[~np.isnan(z_3d)]
      frac_within_2sigma = float((np.abs(z_valid) <= 2).mean() * 100) if len(z_valid) else float("nan")
      frac_within_3sigma = float((np.abs(z_valid) <= 3).mean() * 100) if len(z_valid) else float("nan")
      zscore_summary = {
          "mean_abs_z": (
              float(np.nanmean(np.abs(z_valid)))
              if len(z_valid)
              else float("nan")
          ),
          "frac_within_2sigma_pct": frac_within_2sigma,
          "frac_within_3sigma_pct": frac_within_3sigma,
          "n_valid": int(len(z_valid)),
      }

      fig, axes = plt.subplots(1, 2, figsize=(14, 6))
      im = axes[0].imshow(
          z_map,
          extent=extent_xy,
          origin="lower",
          cmap="RdBu_r",
          vmin=-Z_ABS_MAX_PLOT,
          vmax=Z_ABS_MAX_PLOT,
      )
      axes[0].set_title(
          "Significatività statistica z = ΔD/σ\n"
          f"{field} — {model.upper()} @ Z={depth_cm_slice}cm"
      )
      axes[0].set_xlabel("X (cm)")
      axes[0].set_ylabel("Y (cm)")
      plt.colorbar(im, ax=axes[0], label="z (σ)")

      axes[1].hist(
          z_valid, bins=80, range=(-8, 8), color="tab:blue", alpha=0.8
      )
      axes[1].axvline(2, color="orange", ls="--", label="|z|=2")
      axes[1].axvline(-2, color="orange", ls="--")
      axes[1].axvline(3, color="red", ls="--", label="|z|=3")
      axes[1].axvline(-3, color="red", ls="--")
      axes[1].set_xlabel("z")
      axes[1].set_ylabel("Numero di voxel")
      axes[1].set_title(
          f"Distribuzione z (3D) — {frac_within_2sigma:.1f}% entro 2σ,"
          f" {frac_within_3sigma:.1f}% entro 3σ"
      )
      axes[1].legend()

      plt.tight_layout()
      plt.savefig(out_dir / f"zscore_map_{model}.png", dpi=130)
      plt.close(fig)

  results["zscore_uncertainty"] = zscore_summary
  if ANALYZE_FIELD_EDGES and half_width_cm is not None:
    results["field_edges_by_depth_cm"] = edge_results_by_depth

  z_summary_str = ""
  if zscore_summary is not None:
    z_summary_str = (
        f", z: {zscore_summary['frac_within_2sigma_pct']:.1f}% entro 2σ"
    )
  print(
      f"  [{model.upper()}] {maps_path.name}, {profiles_path.name}, gamma: "
      + ", ".join(
          f"{k}={v['pass_rate_pct']:.1f}%"
          for k, v in results.items()
          if isinstance(v, dict) and "pass_rate_pct" in v
      )
      + z_summary_str
  )
  return results


def main():
  parser = argparse.ArgumentParser(
      description="Report dosimetrico esteso con salvataggio incrementale."
  )
  parser.add_argument(
      "--fields",
      nargs="+",
      default=FIELDS,
      help="Campi da analizzare (default: tutti)",
  )
  parser.add_argument(
      "--skip_existing",
      action="store_true",
      help="Salta i campi gia' presenti con tutti i modelli nel summary JSON",
  )
  args = parser.parse_args()

  base = Path(BASE_DIR)
  summary_path = base / "gamma_summary_with_2d_maps_extended.json"

  summary = {}
  if summary_path.exists():
    try:
      with open(summary_path, "r") as f:
        summary = json.load(f)
      print(
          f"Caricato summary preesistente con {len(summary)} campi:"
          f" {list(summary.keys())}"
      )
    except Exception as e:
      print(f"Lettura summary fallita ({e}), verra' reinizializzato.")
      summary = {}

  for field in args.fields:
    if (
        args.skip_existing
        and field in summary
        and len(summary[field]) >= len(MODELS)
    ):
      print(f"\n⏩ [{field}] gia' completato nel summary, salto.")
      continue

    field_dir = base / field
    ref_path = field_dir / "dose_reference_dose.mhd"
    if not ref_path.exists():
      print(f"❌ [{field}] reference non trovato in {ref_path}, salto.")
      continue

    print(f"\n{'='*70}\n CLASSE: {field}\n{'='*70}")
    ref_arr, spacing_zyx = load_dose(ref_path)
    ref_gnorm = robust_max(ref_arr)
    ref_norm = (ref_arr / ref_gnorm) * 100.0

    out_dir = field_dir
    if field not in summary:
      summary[field] = {}

    for model in MODELS:
      res = process_pair(
          field_dir,
          field,
          model,
          ref_arr,
          ref_norm,
          ref_gnorm,
          spacing_zyx,
          out_dir,
          ref_path,
      )
      if res is not None:
        summary[field][model] = res

    # Salvataggio progressivo su disco dopo ciascun campo completato
    with open(summary_path, "w") as f:
      json.dump(summary, f, indent=2)
    print(f"💾 [Salvataggio incrementale] Aggiornato {summary_path}")

  print(f"\n{'='*70}")
  print(f" Elaborazione conclusa. Riepilogo: {summary_path}")
  print(f"{'='*70}")


if __name__ == "__main__":
  main()
