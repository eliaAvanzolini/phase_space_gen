import os
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")  # headless, niente display sul cluster
import matplotlib.pyplot as plt

BASE_DIR = "outputs/dose_validation"
FIELDS = ["6mv_5x5", "6mv_10x10", "6mv_20x20", "10mv_5x5", "10mv_10x10", "10mv_20x20"]
MODELS = ["cfm", "nsf"]
COLORS = {"reference": "black", "cfm": "tab:blue", "nsf": "tab:orange"}

# Profilo trasversale preso a due profondita': una vicina al buildup (early),
# una piu' stabile in zona di plateau (deep). In cm, verranno convertite in
# indici voxel in base allo spacing reale letto dall'header.
DEPTHS_CM_TO_PLOT = [3.0, 10.0]


def load_dose(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float64)  # ordine (Z, Y, X)
    spacing = img.GetSpacing()  # (X, Y, Z)
    spacing_zyx = spacing[::-1]
    return arr, spacing_zyx


def robust_norm(arr):
    nz = arr[arr > 0]
    p999 = np.percentile(nz, 99.9) if len(nz) > 0 else arr.max()
    return (arr / p999) * 100.0, p999


def plot_field(field):
    field_dir = os.path.join(BASE_DIR, field)
    ref_path = os.path.join(field_dir, "dose_reference_dose.mhd")
    if not os.path.exists(ref_path):
        print(f"❌ [{field}] reference non trovato, salto.")
        return

    ref_arr, spacing_zyx = load_dose(ref_path)
    ref_norm, ref_p999 = robust_norm(ref_arr)
    nz_z, ny, nx = ref_arr.shape
    cz, cy, cx = nz_z // 2, ny // 2, nx // 2  # centro griglia (asse centrale x=0,y=0)

    sp_z, sp_y, sp_x = spacing_zyx  # mm

    # Intorno laterale su cui mediare (in voxel, per lato) per sopprimere il
    # rumore di conteggio per-voxel. Una riga singola di voxel e' dominata dal
    # rumore MC anche col reference: mediando su un piccolo kernel si recupera
    # il segnale "vero" se c'e'.
    HALF_WIDTH_VOX = 3  # media su una finestra di (2*3+1)=7 voxel di lato

    def extract_pdd(norm_arr, cy, cx, hw):
        y0, y1 = max(0, cy - hw), min(norm_arr.shape[1], cy + hw + 1)
        x0, x1 = max(0, cx - hw), min(norm_arr.shape[2], cx + hw + 1)
        return norm_arr[:, y0:y1, x0:x1].mean(axis=(1, 2))

    def extract_transverse(norm_arr, z_idx, cy, hw):
        y0, y1 = max(0, cy - hw), min(norm_arr.shape[1], cy + hw + 1)
        return norm_arr[z_idx, y0:y1, :].mean(axis=0)

    maps = {"reference": (ref_arr, ref_norm)}
    for model in MODELS:
        model_path = os.path.join(field_dir, f"dose_{model}_dose.mhd")
        if not os.path.exists(model_path):
            print(f"  ⚠️ [{model.upper()}] mappa non trovata, salto dal plot.")
            continue
        m_arr, m_sp = load_dose(model_path)
        if m_arr.shape != ref_arr.shape:
            print(f"  ⚠️ [{model.upper()}] shape diversa da reference ({m_arr.shape} vs {ref_arr.shape}), salto.")
            continue
        m_norm, _ = robust_norm(m_arr)
        maps[model] = (m_arr, m_norm)

    n_depth_plots = len(DEPTHS_CM_TO_PLOT)
    fig, axes = plt.subplots(1, 1 + n_depth_plots, figsize=(6 * (1 + n_depth_plots), 5))

    # ── PDD: profilo lungo l'asse centrale (Z), a x=0,y=0 ──────────────────
    ax_pdd = axes[0]
    z_axis_mm = np.arange(nz_z) * sp_z
    for name, (raw, norm) in maps.items():
        profile = extract_pdd(norm, cy, cx, HALF_WIDTH_VOX)
        ax_pdd.plot(z_axis_mm / 10.0, profile, label=name.upper(), color=COLORS.get(name, None))
    ax_pdd.set_xlabel("Profondita' Z (cm)")
    ax_pdd.set_ylabel(f"Dose relativa (% P99.9), media su {2*HALF_WIDTH_VOX+1}x{2*HALF_WIDTH_VOX+1} voxel")
    ax_pdd.set_title(f"{field} — PDD (asse centrale, mediato)")
    ax_pdd.legend()
    ax_pdd.grid(alpha=0.3)

    # ── Profili trasversali a profondita' fissate ──────────────────────────
    x_axis_mm = (np.arange(nx) - cx) * sp_x
    for i, depth_cm in enumerate(DEPTHS_CM_TO_PLOT):
        ax = axes[1 + i]
        z_idx = int(round((depth_cm * 10.0) / sp_z))
        z_idx = max(0, min(z_idx, nz_z - 1))
        for name, (raw, norm) in maps.items():
            profile = extract_transverse(norm, z_idx, cy, HALF_WIDTH_VOX)
            ax.plot(x_axis_mm / 10.0, profile, label=name.upper(), color=COLORS.get(name, None))
        ax.set_xlabel("Posizione X (cm)")
        ax.set_ylabel(f"Dose relativa (% P99.9), media su {2*HALF_WIDTH_VOX+1} voxel in Y")
        ax.set_title(f"{field} — profilo trasversale @ Z={depth_cm}cm (mediato)")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(field_dir, "profile_comparison.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"✅ [{field}] salvato -> {out_path}")


def main():
    for field in FIELDS:
        plot_field(field)


if __name__ == "__main__":
    main()
