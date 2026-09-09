import h5py
import numpy as np
import uproot

REF_H5 = "data/energy_only_10mv_reference.h5"
GEN_REF_ROOT = "outputs/dose_interpolation_10mv/gen_reference.root"
GEN_CFM_ROOT = "outputs/dose_interpolation_10mv/gen_cfm.root"
N_SAMPLE = 2_000_000
N_RADIAL_BINS = 20
MAX_R_CM = 10.0


def radial_energy_profile(x_cm, y_cm, E, n_bins, max_r):
  r = np.sqrt(x_cm**2 + y_cm**2)
  bin_edges = np.linspace(0, max_r, n_bins + 1)
  centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
  mean_E = np.full(n_bins, np.nan)
  n_per_bin = np.zeros(n_bins, dtype=int)
  for i in range(n_bins):
    m = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
    n_per_bin[i] = m.sum()
    if m.sum() > 50:
      mean_E[i] = E[m].mean()
  return centers, mean_E, n_per_bin


print("=" * 70)
print(" 1. STRUTTURA COLONNE DEL DATASET REFERENCE")
print("=" * 70)
with h5py.File(REF_H5, "r") as f:
  shape = f["phase_space"].shape
  print(f"  Shape phase_space: {shape}  (n_particelle, n_colonne)")
  print(
      "  Attrs del dataset:"
      f" {dict(f['phase_space'].attrs) if f['phase_space'].attrs else '(nessuno)'}"
  )
  print(f"  Chiavi nel gruppo/file: {list(f.keys())}")

  if shape[1] > 2:
    col2 = f["phase_space"][:200_000, 2]
    uniq = np.unique(col2)
    print("\n  Colonna indice 2 (non usata da write_phsp_root!):")
    print(
        f"    min={col2.min():.4g} max={col2.max():.4g}"
        f" n_valori_unici={len(uniq)}"
    )
    if len(uniq) <= 10:
      print(
          f"    Valori unici: {uniq}  <-- se questo e' un codice tipo particella"
          " (es. 0=gamma, 1=e-, 2=e+), l'informazione originale è stata persa"
          " forzando tutto a gamma"
      )
    else:
      print(
          "    Continua (probabile coordinata Z o peso statistico, non tipo"
          " particella)"
      )

print()
print("=" * 70)
print(" 2. CORRELAZIONE ENERGIA-RAGGIO: reference vs CFM")
print("=" * 70)
with uproot.open(GEN_REF_ROOT) as f:
  key = [k for k in f.keys() if "PhaseSpace" in k][0]
  arr = f[key].arrays(["X", "Y", "E"], library="np")
  x_ref, y_ref, E_ref = (
      arr["X"][:N_SAMPLE] / 10.0,
      arr["Y"][:N_SAMPLE] / 10.0,
      arr["E"][:N_SAMPLE],
  )

with uproot.open(GEN_CFM_ROOT) as f:
  key = [k for k in f.keys() if "PhaseSpace" in k][0]
  arr = f[key].arrays(["X", "Y", "E"], library="np")
  x_cfm, y_cfm, E_cfm = (
      arr["X"][:N_SAMPLE] / 10.0,
      arr["Y"][:N_SAMPLE] / 10.0,
      arr["E"][:N_SAMPLE],
  )

r_ref, meanE_ref, n_ref = radial_energy_profile(
    x_ref, y_ref, E_ref, N_RADIAL_BINS, MAX_R_CM
)
r_cfm, meanE_cfm, n_cfm = radial_energy_profile(
    x_cfm, y_cfm, E_cfm, N_RADIAL_BINS, MAX_R_CM
)

print(
    f"  {'raggio(cm)':>10s} {'E_ref(MeV)':>11s} {'E_cfm(MeV)':>11s}"
    f" {'diff(%)':>9s} {'n_ref':>10s} {'n_cfm':>10s}"
)
for i in range(N_RADIAL_BINS):
  diff_pct = (
      100 * (meanE_cfm[i] - meanE_ref[i]) / meanE_ref[i]
      if meanE_ref[i]
      else float("nan")
  )
  flag = "  <-- ZONA ANELLO" if 6.0 <= r_ref[i] <= 8.5 else ""
  print(
      f"  {r_ref[i]:10.2f} {meanE_ref[i]:11.3f} {meanE_cfm[i]:11.3f}"
      f" {diff_pct:8.1f}% {n_ref[i]:10d} {n_cfm[i]:10d}{flag}"
  )

print()
print("=" * 70)
print(" 3. FRAZIONE DI ENERGIE NON FISICHE NEL CFM (da filtrare)")
print("=" * 70)
n_neg = int((E_cfm <= 0).sum())
n_low = int(((E_cfm > 0) & (E_cfm < 0.01)).sum())
print(
    f"  E <= 0:            {n_neg:,} / {len(E_cfm):,} "
    f" ({100*n_neg/len(E_cfm):.3f}%)"
)
print(
    f"  0 < E < 0.01 MeV:  {n_low:,} / {len(E_cfm):,} "
    f" ({100*n_low/len(E_cfm):.3f}%)"
)
print(
    "  Totale da filtrare (E <= 0.01):"
    f" {100*(n_neg+n_low)/len(E_cfm):.3f}%"
)
