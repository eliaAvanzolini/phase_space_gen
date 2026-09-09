import json
from pathlib import Path
import numpy as np
import pymedphys
import SimpleITK as sitk

BASE_DIR = Path("outputs/dose_interpolation_10mv")
OUT_DIR = Path("outputs/gamma_bulk_vs_surface")

BUILDUP_CUTOFF_CM = 1.0  # Z < 1.0 cm = superficie/build-up, Z >= 1.0 cm = bulk
CRITERIA = [
    {"name": "3pct_3mm", "dose_pct": 3.0, "dist_mm": 3.0},
    {"name": "2pct_2mm_clinico", "dose_pct": 2.0, "dist_mm": 2.0},
]
LOWER_DOSE_CUTOFF = 10.0


def load_dose(path):
  img = sitk.ReadImage(str(path))
  arr = sitk.GetArrayFromImage(img).astype(np.float64)
  spacing_zyx = img.GetSpacing()[::-1]
  return arr, spacing_zyx


def robust_max(arr, cy, cx, hw=3):
  y0, y1 = max(0, cy - hw), min(arr.shape[1], cy + hw + 1)
  x0, x1 = max(0, cx - hw), min(arr.shape[2], cx + hw + 1)
  pdd_roi = arr[:, y0:y1, x0:x1].mean(axis=(1, 2))
  return float(pdd_roi.max())


def gamma_pass_rate(
    ref_arr, model_arr, spacing_zyx, ref_gnorm, z_mask_3d, crit
):
  axes_3d = tuple(np.arange(s) * sp for s, sp in zip(ref_arr.shape, spacing_zyx))
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
  valid = gamma_map[z_mask_3d & ~np.isnan(gamma_map)]
  pass_rate = float((valid <= 1.0).mean() * 100) if len(valid) else float("nan")
  return pass_rate, int(len(valid))


def main():
  OUT_DIR.mkdir(parents=True, exist_ok=True)

  ref_arr, spacing_zyx = load_dose(BASE_DIR / "dose_reference_dose.mhd")
  model_arr, _ = load_dose(BASE_DIR / "dose_cfm_dose.mhd")
  sp_z, sp_y, sp_x = spacing_zyx
  nz, ny, nx = ref_arr.shape
  cy, cx = ny // 2, nx // 2

  ref_gnorm = robust_max(ref_arr, cy, cx)

  z_axis_cm = (np.arange(nz) * sp_z / 10.0)[:, None, None]
  surface_mask = np.broadcast_to(z_axis_cm < BUILDUP_CUTOFF_CM, ref_arr.shape)
  bulk_mask = ~surface_mask

  print("=" * 70)
  print(f" ANALISI GAMMA: BULK (Z >= {BUILDUP_CUTOFF_CM}cm) vs SUPERFICIE")
  print("=" * 70)

  results = {"cutoff_cm": BUILDUP_CUTOFF_CM}
  for region_name, mask in [
      ("full_volume", np.ones_like(ref_arr, dtype=bool)),
      ("surface_buildup", surface_mask),
      ("bulk_excl_buildup", bulk_mask),
  ]:
    results[region_name] = {}
    for crit in CRITERIA:
      pr, n = gamma_pass_rate(
          ref_arr, model_arr, spacing_zyx, ref_gnorm, mask, crit
      )
      results[region_name][crit["name"]] = {"pass_rate_pct": pr, "n_valid": n}
      print(
          f" [{region_name:20s}] {crit['name']:20s} pass rate = {pr:5.1f}% "
          f" (n={n:,})"
      )

  out_file = OUT_DIR / "gamma_bulk_vs_surface.json"
  with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

  print("=" * 70)
  print(f" Salvato riepilogo in: {out_file}")


if __name__ == "__main__":
  main()
