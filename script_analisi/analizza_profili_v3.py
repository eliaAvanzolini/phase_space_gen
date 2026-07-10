import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys

base = "outputs/outputs_condizionali_6mv_5x5/run_scale_test"

def load_dose_matrix(pattern):
    files = sorted(glob.glob(f"{base}/{pattern}"))
    if not files:
        print(f"❌ Errore: impossibile trovare file per '{pattern}'")
        return None
    return sitk.GetArrayFromImage(sitk.ReadImage(files[0])).astype(np.float64)

print("=========================================================")
print("🔬 ANALISI DOSIMETRICA REALE CON FIX BUILD-UP E ROW-MEAN")
print("=========================================================")

ref = load_dose_matrix("dose_reference_nativo*.mhd")
cfm_118k = load_dose_matrix("dose_cfm_118k*.mhd")
cfm_1m = load_dose_matrix("dose_cfm_1M*.mhd")

if ref is None or cfm_118k is None or cfm_1m is None:
    print("❌ File mancanti. Interruzione.")
    sys.exit(1)

# 1. INDIVIDUAZIONE DEL VERO ASSE E VOXEL DI BUILD-UP (Media su X,Y)
# In SimpleITK, ref.shape è (Z, Y, X). Facciamo la media su Y e X per ogni fetta Z.
depth_profile = ref.mean(axis=(1, 2))
z_peak = np.argmax(depth_profile)

print(f"📊 Configurazione griglia: Z={ref.shape[0]}, Y={ref.shape[1]}, X={ref.shape[2]}")
print(f"🎯 Vero voxel di Build-up individuato lungo Z: {z_peak} (Profondità reale: {z_peak * 4.0} mm)")

# 2. NORMALIZZAZIONE AL PICCO REALE DELLA DOSE MEDIATA
# Per evitare che un singolo pixel isolato distorca tutto, normalizziamo stabiizzandoci sul massimo del profilo centrale
ref_norm = (ref / ref[z_peak, :, :].max()) * 100.0
cfm_118k_norm = (cfm_118k / cfm_118k[z_peak, :, :].max()) * 100.0
cfm_1m_norm = (cfm_1m / cfm_1m[z_peak, :, :].max()) * 100.0

# 3. ESTRAZIONE DEI PROFILI TRASVERSALI CON MEDIA SU UNA FASCIA DI RIGHE (Y_center +/- 3 voxel)
y_c = ref.shape[1] // 2
profile_ref = ref_norm[z_peak, y_c-3:y_c+4, :].mean(axis=0)
profile_118k = cfm_118k_norm[z_peak, y_c-3:y_c+4, :].mean(axis=0)
profile_1m = cfm_1m_norm[z_peak, y_c-3:y_c+4, :].mean(axis=0)

# Asse spaziale in millimetri
voxel_mm = 4.0
x_axis = (np.arange(len(profile_ref)) - len(profile_ref) / 2) * voxel_mm

# --- COSTRUZIONE DEL GRAFICO CLINICO REALE ---
plt.figure(figsize=(10, 6))
plt.plot(x_axis, profile_ref, label="Reference Nativo (118k storie)", linewidth=2.0, color="black", alpha=0.8)
plt.plot(x_axis, profile_118k, label="CFM Baseline 1:1 (118k storie)", linewidth=1.5, linestyle=":", color="crimson")
plt.plot(x_axis, profile_1m, label="CFM Alta Statistica (1M storie)", linewidth=2.5, linestyle="--", color="forestgreen")

plt.xlabel("Posizione X [mm]", fontsize=12)
plt.ylabel("Dose Normalizzata [%]", fontsize=12)
plt.title(f"Profilo Trasversale Pulito al Vero Picco di Build-up (Z Voxel = {z_peak})", fontsize=13, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=11, loc="lower center")
plt.xlim(-60, 60)

plot_path = f"{base}/profile_comparison_clean.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"🎨 ✅ Nuovo grafico pulito salvato in: {plot_path}")

# --- CALCOLO DELLA LARGHEZZA DI PENOMBRA (FWHM) ---
def field_width_at_level(profile, x_axis, level=50.0):
    above = profile >= level
    if not above.any(): return None
    idx = np.where(above)[0]
    return x_axis[idx[-1]] - x_axis[idx[0]]

w_ref = field_width_at_level(profile_ref, x_axis)
w_118k = field_width_at_level(profile_118k, x_axis)
w_1m = field_width_at_level(profile_1m, x_axis)

print("\n📊 NUOVO VERDETTO LARGHEZZE DI CAMPO PURIFICATE (FWHM):")
print("-" * 60)
print(f" 🟢 Width REFERENCE: {w_ref:.1f} mm")
print(f" 🔴 Width CFM 118k:  {w_118k:.1f} mm  |  Δ vs Ref = {w_118k - w_ref:+.1f} mm")
print(f" 🍏 Width CFM 1M:    {w_1m:.1f} mm  |  Δ vs Ref = {w_1m - w_ref:+.1f} mm")
print("-" * 60)
print("=========================================================\n")
