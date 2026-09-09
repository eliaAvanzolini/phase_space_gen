from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

BASE_DIR = "outputs/dose_interpolation_10mv"
DEPTHS_CM = [0.0, 1.0, 3.0, 5.0]
N_RADIAL_BINS = 60
MAX_RADIUS_CM = 10.0


def load_dose(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float64)
    spacing_zyx = img.GetSpacing()[::-1]
    return arr, spacing_zyx


def find_uncertainty_path(dose_path: Path):
    stem = dose_path.name
    for cand in [dose_path.with_name(stem.replace("_dose.mhd", "_dose_uncertainty.mhd"))]:
        if cand.exists():
            return cand
    run_tag = stem.split("_dose")[0]
    for p in dose_path.parent.glob(f"*{run_tag}*uncertain*.mhd"):
        return p
    return None


def radial_profile(arr_2d, cy, cx, sp_x, sp_y, n_bins, max_r_cm):
    ny, nx = arr_2d.shape
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    r_cm = np.sqrt(((xx - cx) * sp_x / 10.0) ** 2 + ((yy - cy) * sp_y / 10.0) ** 2)
    bin_edges = np.linspace(0, max_r_cm, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    means = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (r_cm >= bin_edges[i]) & (r_cm < bin_edges[i + 1])
        if m.sum() > 0:
            means[i] = arr_2d[m].mean()
    return bin_centers, means


def main():
    base = Path(BASE_DIR)
    ref_path = base / "dose_reference_dose.mhd"
    cfm_path = base / "dose_cfm_dose.mhd"

    if not ref_path.exists() or not cfm_path.exists():
        raise FileNotFoundError(f"File dose non trovati in {base}")

    print(f"Caricamento matrici da {base}...")
    ref_arr, spacing_zyx = load_dose(ref_path)
    cfm_arr, _ = load_dose(cfm_path)
    sp_z, sp_y, sp_x = spacing_zyx
    nz, ny, nx = ref_arr.shape
    cy, cx = ny // 2, nx // 2

    unc_ref_p = find_uncertainty_path(ref_path)
    unc_cfm_p = find_uncertainty_path(cfm_path)
    unc_ref = load_dose(unc_ref_p)[0] * ref_arr if unc_ref_p else None
    unc_cfm = load_dose(unc_cfm_p)[0] * cfm_arr if unc_cfm_p else None

    fig, axes = plt.subplots(2, len(DEPTHS_CM), figsize=(5.5 * len(DEPTHS_CM), 9))

    for j, depth_cm in enumerate(DEPTHS_CM):
        z_idx = max(0, min(int(round(depth_cm * 10.0 / sp_z)), nz - 1))

        r_ref, prof_ref = radial_profile(ref_arr[z_idx], cy, cx, sp_x, sp_y, N_RADIAL_BINS, MAX_RADIUS_CM)
        _, prof_cfm = radial_profile(cfm_arr[z_idx], cy, cx, sp_x, sp_y, N_RADIAL_BINS, MAX_RADIUS_CM)

        norm = np.nanmax(prof_ref)
        axes[0, j].plot(r_ref, prof_ref / norm * 100, label="REFERENCE", color="black", lw=1.5)
        axes[0, j].plot(r_ref, prof_cfm / norm * 100, label="CFM", color="tab:blue", lw=1.5)
        axes[0, j].set_title(f"Profilo radiale @ Z={depth_cm}cm")
        axes[0, j].set_xlabel("Raggio (cm)")
        axes[0, j].set_ylabel("Dose (% picco radiale ref)")
        axes[0, j].legend()
        axes[0, j].grid(alpha=0.3)

        if unc_ref is not None and unc_cfm is not None:
            _, unc_ref_r = radial_profile(unc_ref[z_idx], cy, cx, sp_x, sp_y, N_RADIAL_BINS, MAX_RADIUS_CM)
            _, unc_cfm_r = radial_profile(unc_cfm[z_idx], cy, cx, sp_x, sp_y, N_RADIAL_BINS, MAX_RADIUS_CM)
            denom = np.sqrt(unc_ref_r ** 2 + unc_cfm_r ** 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                z_r = np.where(denom > 0, (prof_cfm - prof_ref) / denom, np.nan)
            axes[1, j].plot(r_ref, z_r, color="tab:red", lw=1.5)
            axes[1, j].axhline(0, color="grey", lw=0.8)
            axes[1, j].axhline(2, color="orange", ls="--", lw=1, label="+2σ")
            axes[1, j].axhline(-2, color="orange", ls="--", lw=1, label="-2σ")
            axes[1, j].set_title(f"z radiale @ Z={depth_cm}cm")
            axes[1, j].set_xlabel("Raggio (cm)")
            axes[1, j].set_ylabel("z = ΔD/σ (media anulare)")
            axes[1, j].grid(alpha=0.3)

    plt.tight_layout()
    out_path = base / "radial_ring_profile.png"
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"✅ Grafico generato con successo: {out_path}")


if __name__ == "__main__":
    main()
