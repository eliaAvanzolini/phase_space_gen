#!/usr/bin/env python3
"""Correct metadata for 30MeV_05cm.phsp when z=-0.5 cm confirms depth=0.5 cm.

Backs up both manifests and the affected HDF5 shard before editing them.
Does NOT reread the original ASCII PHSP.
"""
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import h5py

ROOT = Path.cwd()
RAW_MANIFEST = ROOT / "data/metadata/manifest_multienergy.csv"
PROC_DIR = ROOT / "data/processed_multienergy"
PROC_MANIFEST = PROC_DIR / "dataset_manifest.csv"
PHSP_BASENAME = "30MeV_05cm.phsp"
EXPECTED_E0 = 30.0
EXPECTED_Z = -0.5
OLD_DEPTH = 5.0
NEW_DEPTH = 0.5


def unique_row(df, col, value, label):
    if col not in df:
        raise RuntimeError(f"Colonna {col!r} assente in {label}")
    mask = df[col].astype(str).map(lambda x: Path(x).name == value)
    idx = df.index[mask].tolist()
    if len(idx) != 1:
        raise RuntimeError(f"Attesa 1 riga per {value} in {label}; trovate {len(idx)}")
    return idx[0]


def backup(path: Path):
    b = path.with_name(path.name + ".bak_depthfix")
    if b.exists():
        raise RuntimeError(f"Backup gia' esistente: {b}; non sovrascrivo nulla")
    shutil.copy2(path, b)
    print(f"Backup: {b}")


def main():
    for p in (RAW_MANIFEST, PROC_MANIFEST):
        if not p.exists():
            raise SystemExit(f"File non trovato: {p}")

    raw = pd.read_csv(RAW_MANIFEST, dtype=str, keep_default_na=False)
    proc = pd.read_csv(PROC_MANIFEST, dtype=str, keep_default_na=False)
    ir = unique_row(raw, "phsp_file", PHSP_BASENAME, str(RAW_MANIFEST))
    ip = unique_row(proc, "phsp_file", PHSP_BASENAME, str(PROC_MANIFEST))

    for label, df, i in (("manifest raw", raw, ir), ("manifest processed", proc, ip)):
        e0 = float(df.loc[i, "beam_energy_MeV"])
        d = float(df.loc[i, "depth_cm"])
        print(f"{label}: E0={e0:g}, depth attuale={d:g}")
        if not np.isclose(e0, EXPECTED_E0):
            raise RuntimeError(f"E0 inattesa in {label}: {e0}")
        if not np.isclose(d, OLD_DEPTH):
            raise RuntimeError(f"La profondita' in {label} non e' {OLD_DEPTH:g} cm: {d:g}; interrompo")

    if "processed_file" not in proc:
        raise RuntimeError("Colonna processed_file assente dal manifest processed")
    rel_h5 = proc.loc[ip, "processed_file"]
    h5path = (PROC_DIR / rel_h5).resolve()
    try:
        h5path.relative_to(PROC_DIR.resolve())
    except ValueError:
        raise RuntimeError(f"Percorso HDF5 non sicuro: {h5path}")
    if not h5path.exists():
        raise RuntimeError(f"HDF5 non trovato: {h5path}")

    # Preflight: check stored metadata and z plane before writing any file.
    with h5py.File(h5path, "r") as h:
        if "conditions" not in h or "phase_space" not in h:
            raise RuntimeError("HDF5 non contiene phase_space e conditions")
        if not np.isclose(float(h.attrs.get("beam_energy_MeV", np.nan)), EXPECTED_E0):
            raise RuntimeError("Attributo beam_energy_MeV inatteso nell'HDF5")
        if not np.isclose(float(h.attrs.get("depth_cm", np.nan)), OLD_DEPTH):
            raise RuntimeError("Attributo depth_cm non e' 5.0 come previsto; interrompo")
        ps, cond = h["phase_space"], h["conditions"]
        if ps.shape[1] < 7 or cond.shape[1] < 2 or len(ps) != len(cond):
            raise RuntimeError("Shape HDF5 inattese")
        z = ps[:, 2]
        if not np.allclose(z, EXPECTED_Z, atol=1e-5, rtol=0):
            raise RuntimeError(f"La coordinata z non e' costante a {EXPECTED_Z}: min={z.min()}, max={z.max()}")
        # Validate current conditions in chunks to keep memory bounded.
        for lo in range(0, len(cond), 250_000):
            c = cond[lo:lo+250_000]
            if not np.allclose(c[:, 0], OLD_DEPTH, atol=1e-5, rtol=0):
                raise RuntimeError("conditions[:,0] non e' tutta a 5.0; interrompo")
            if not np.allclose(c[:, 1], EXPECTED_E0, atol=1e-5, rtol=0):
                raise RuntimeError("conditions[:,1] non e' tutta a 30 MeV; interrompo")
        n = len(ps)

    print(f"\nControlli superati per {h5path.name}: N={n:,}, z={EXPECTED_Z:g} cm.")
    print("Interpreto '05cm' come 0.5 cm, coerentemente con z=-0.5 cm.")
    answer = input("Procedo aggiornando HDF5 e manifest? Scrivi YES per confermare: ").strip()
    if answer != "YES":
        print("Annullato; nessun file modificato.")
        return

    # Backups before edits.
    backup(RAW_MANIFEST); backup(PROC_MANIFEST); backup(h5path)

    with h5py.File(h5path, "r+") as h:
        cond = h["conditions"]
        for lo in range(0, len(cond), 250_000):
            hi = min(len(cond), lo + 250_000)
            block = cond[lo:hi]
            block[:, 0] = np.float32(NEW_DEPTH)
            cond[lo:hi] = block
        h.attrs["depth_cm"] = NEW_DEPTH
        h.attrs["depth_correction_note"] = "Filename 30MeV_05cm; corrected from 5.0 to 0.5 cm because phase_space z is -0.5 cm. Verify against simulation metadata."

    note = "Depth corrected from 5.0 to 0.5 cm: HDF5 phase_space z is constant at -0.5 cm; confirm with simulation metadata."
    for df, i in ((raw, ir), (proc, ip)):
        df.loc[i, "depth_cm"] = str(NEW_DEPTH)
        if "notes" in df:
            prev = df.loc[i, "notes"].strip()
            df.loc[i, "notes"] = (prev + " | " if prev else "") + note
    raw.to_csv(RAW_MANIFEST, index=False)
    proc.to_csv(PROC_MANIFEST, index=False)

    # Final read-back validation.
    with h5py.File(h5path, "r") as h:
        assert np.isclose(float(h.attrs["depth_cm"]), NEW_DEPTH)
        assert np.allclose(h["conditions"][:1000, 0], NEW_DEPTH)
        assert np.allclose(h["conditions"][:1000, 1], EXPECTED_E0)
    print("\nCORREZIONE COMPLETATA")
    print(f"  shard: {h5path}")
    print(f"  depth_cm attr / conditions[:,0] -> {NEW_DEPTH}")
    print(f"  manifest aggiornati: {RAW_MANIFEST}, {PROC_MANIFEST}")
    print("  backup *.bak_depthfix conservati. Non occorre ricostruire lo shard.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        sys.exit(2)
