#!/usr/bin/env python3
"""Read and inspect TOPAS ASCII phase-space files (.phsp + matching .header).

Column order in the TOPAS ASCII file:
  x, y, z, dx, dy, E, weight, PDG, dz_negative_flag, first_particle_of_history_flag
The 7D model array uses (x, y, z, dx, dy, dz, E). The original metadata columns
(weight, PDG, flags) are retained separately by the conversion/build scripts.

Filename inference is deliberately conservative. Supported examples include:
  70MeV_3cm.phsp, beam70MeV_depth3cm.phsp, E0_70MeV_water_3.5cm.phsp,
  SurfaceFile_3_5_peak_1_2_5.phsp (depth/region inferred; E0 remains unknown).
If E0 or depth cannot be inferred, provide it explicitly in the manifest.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import pandas as pd

COLS = ["x", "y", "z", "dx", "dy", "E", "w", "pdg", "dz_neg", "first"]
PROTON = 2212
PDG_NAMES = {2212: "proton", 11: "e-", -11: "e+", 22: "gamma", 2112: "neutron",
             1000010020: "deuteron", 1000020040: "alpha"}


def paired_header(phsp_path: str | Path) -> Path:
    return Path(phsp_path).with_suffix(".header")


def read_header(phsp_path: str | Path) -> Tuple[Dict, str]:
    """Return parsed header values and full raw header text."""
    hp = paired_header(phsp_path)
    if not hp.exists():
        return {}, ""
    txt = hp.read_text(errors="replace")
    info: Dict = {"header_file": str(hp)}
    patterns = {
        "n_histories": r"Number of Original Histories:\s*(\d+)",
        "n_histories_reached": r"Number of Original Histories that Reached Phase Space:\s*(\d+)",
        "n_particles": r"Number of Scored Particles:\s*(\d+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, txt, flags=re.I)
        if m:
            info[key] = int(m.group(1))
    for m in re.finditer(r"Number of\s+([^:\n]+):\s*(\d+)\s*$", txt, flags=re.I | re.M):
        species = re.sub(r"\s+", " ", m.group(1).strip().lower())
        # These are summary counters, not particle species.
        if "original histor" in species or "scored particle" in species:
            continue
        info["species_" + re.sub(r"[^a-z0-9]+", "_", species).strip("_")] = int(m.group(2))
    for kind, pat in (("min_energy", r"Minimum Kinetic Energy of\s+([^:]+):\s*([+\-0-9.eE]+)\s*MeV"),
                       ("max_energy", r"Maximum Kinetic Energy of\s+([^:]+):\s*([+\-0-9.eE]+)\s*MeV")):
        for m in re.finditer(pat, txt, flags=re.I):
            species = re.sub(r"\s+", " ", m.group(1).strip().lower())
            key = re.sub(r"[^a-z0-9]+", "_", species).strip("_")
            info[f"{kind}_{key}_MeV"] = float(m.group(2))
    return info, txt


def parse_conditions_from_name(filename: str | Path) -> Dict[str, Optional[float | str]]:
    """Conservative filename parser. Returns E0_MeV, depth_cm, region."""
    name = Path(filename).name
    stem = Path(name).stem
    e0: Optional[float] = None
    depth: Optional[float] = None
    region: Optional[str] = None

    # E.g. 70MeV, beam_70MeV, E0_70MeV. Explicit prefixes prevent E0_70MeV
    # from being misread as the decimal number 0.70 MeV.
    em = re.search(r"(?i)(?:beam|e0|energy)[_-]?(\d+(?:[._]\d+)?)\s*MeV(?![A-Za-z])", stem)
    if em is None:
        em = re.search(r"(?i)(?<![A-Za-z0-9])(\d+(?:[._]\d+)?)\s*MeV(?![A-Za-z])", stem)
    if em:
        e0 = float(em.group(1).replace("_", "."))

    rm = re.search(r"(?i)(?:^|[_-])(peak|valley)(?:[_-]|$)", stem)
    if rm:
        region = rm.group(1).lower()

    # Old project naming: SurfaceFile_3.5_peak_... or SurfaceFile_3_5_peak_...
    dm = re.search(r"(?i)SurfaceFile[_-](\d+(?:[._]\d+)?)_(?:peak|valley)(?:[_-]|$)", stem)
    if dm:
        depth = float(dm.group(1).replace("_", "."))
    else:
        # Prefer explicit depth/water tags.
        dm = re.search(r"(?i)(?:depth|water|d)[_-]?(\d+(?:[._]\d+)?)\s*cm?\b", stem)
        if dm:
            depth = float(dm.group(1).replace("_", "."))
        else:
            # General filenames such as 70MeV_3cm; if several cm values exist,
            # the last one is usually the scoring depth, but verify the manifest.
            cms = list(re.finditer(r"(?i)(\d+(?:[._]\d+)?)\s*cm\b", stem))
            if cms:
                depth = float(cms[-1].group(1).replace("_", "."))
    return {"beam_energy_MeV": e0, "depth_cm": depth, "region": region}


def iter_phsp_chunks(path: str | Path, chunksize: int = 250_000) -> Iterator[pd.DataFrame]:
    """Read ASCII rows in chunks to avoid holding an entire simulation in RAM."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    reader = pd.read_csv(path, sep=r"\s+", header=None, names=COLS, engine="c",
                         dtype={"pdg": "int64", "dz_neg": "int8", "first": "int8"},
                         chunksize=chunksize)
    yield from reader


def to_7d(df: pd.DataFrame) -> Tuple[np.ndarray, int]:
    """Convert a chunk to float32 (x,y,z,dx,dy,dz,E), returning invalid direction count."""
    dx = df["dx"].to_numpy(dtype=np.float64)
    dy = df["dy"].to_numpy(dtype=np.float64)
    dz2 = 1.0 - dx * dx - dy * dy
    invalid = int(np.count_nonzero(dz2 < -1e-6))
    dz = np.sqrt(np.clip(dz2, 0.0, 1.0))
    dz = np.where(df["dz_neg"].to_numpy(dtype=np.int8) == 1, -dz, dz)
    arr = np.column_stack([
        df["x"].to_numpy(dtype=np.float32),
        df["y"].to_numpy(dtype=np.float32),
        df["z"].to_numpy(dtype=np.float32),
        dx.astype(np.float32), dy.astype(np.float32), dz.astype(np.float32),
        df["E"].to_numpy(dtype=np.float32),
    ])
    return arr, invalid


def inspect_file(path: str | Path, chunksize: int = 250_000) -> Dict:
    """Stream through a phsp and report counts and proton marginal statistics."""
    path = Path(path)
    hdr, header_text = read_header(path)
    cond = parse_conditions_from_name(path)
    n_rows = n_protons = n_invalid = 0
    species: Dict[str, int] = {}
    vals = {k: [] for k in ("x", "y", "dx", "dy", "E")}
    # For concise stats, retain at most ~500k proton samples for quantile stats;
    # means and stds are computed exactly in streaming accumulators below.
    sums = {k: 0.0 for k in vals}; sums2 = {k: 0.0 for k in vals}; nstat = 0
    for chunk in iter_phsp_chunks(path, chunksize):
        n_rows += len(chunk)
        vc = chunk["pdg"].value_counts()
        for pdg, num in vc.items():
            label = PDG_NAMES.get(int(pdg), f"pdg_{int(pdg)}")
            species[label] = species.get(label, 0) + int(num)
        pr = chunk[chunk["pdg"] == PROTON]
        if len(pr):
            n_protons += len(pr)
            x = pr[["x", "y", "dx", "dy", "E"]].to_numpy(dtype=np.float64)
            nstat += len(x)
            for j, k in enumerate(vals):
                sums[k] += float(x[:, j].sum())
                sums2[k] += float(np.square(x[:, j]).sum())
                vals[k].append((float(x[:, j].min()), float(x[:, j].max())))
        _, invalid = to_7d(chunk)
        n_invalid += invalid
    stats = {}
    for k in vals:
        mu = sums[k] / max(nstat, 1)
        var = max(sums2[k] / max(nstat, 1) - mu * mu, 0.0)
        stats[k] = {"mean": mu, "std_population": float(np.sqrt(var)),
                    "min": min((p[0] for p in vals[k]), default=float("nan")),
                    "max": max((p[1] for p in vals[k]), default=float("nan"))}
    return {"phsp_file": str(path), "header_file": str(paired_header(path)),
            "header_exists": paired_header(path).exists(), **cond,
            "n_rows_actual": n_rows, "n_protons_actual": n_protons,
            "species_actual": species, "invalid_direction_rows": n_invalid,
            "header": hdr, "proton_stats": stats, "header_text": header_text}


def write_one_h5(phsp_path: str | Path, out_path: str | Path, beam_energy_MeV: Optional[float],
                 depth_cm: Optional[float], protons_only: bool = True, chunksize: int = 250_000) -> Dict:
    """Create a single-simulation HDF5 shard, preserving TOPAS metadata columns."""
    import h5py
    phsp_path, out_path = Path(phsp_path), Path(out_path)
    hdr, header_text = read_header(phsp_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_input = n_output = n_protons = n_invalid = 0
    e_min, e_max = np.inf, -np.inf
    with h5py.File(out_path, "w") as h:
        def ds(name, shape, maxshape, dtype, chunk_shape):
            return h.create_dataset(name, shape=shape, maxshape=maxshape, dtype=dtype,
                                    chunks=chunk_shape, compression="gzip", compression_opts=4, shuffle=True)
        ch = max(1, min(chunksize, 65_536))
        ps_ds = ds("phase_space", (0, 7), (None, 7), "f4", (ch, 7))
        cond_ds = ds("conditions", (0, 2), (None, 2), "f4", (ch, 2))
        w_ds = ds("weights", (0,), (None,), "f4", (ch,))
        pdg_ds = ds("pdg", (0,), (None,), "i4", (ch,))
        dzneg_ds = ds("dz_neg", (0,), (None,), "i1", (ch,))
        first_ds = ds("first", (0,), (None,), "i1", (ch,))
        for chunk in iter_phsp_chunks(phsp_path, chunksize):
            n_input += len(chunk)
            n_protons += int((chunk["pdg"] == PROTON).sum())
            if protons_only:
                chunk = chunk.loc[chunk["pdg"] == PROTON]
            if len(chunk) == 0:
                continue
            ps, invalid = to_7d(chunk)
            n_invalid += invalid
            E = chunk["E"].to_numpy(dtype=np.float64)
            e_min = min(e_min, float(E.min())); e_max = max(e_max, float(E.max()))
            n = len(chunk); old = n_output; new = old + n
            for dsx in (ps_ds, cond_ds, w_ds, pdg_ds, dzneg_ds, first_ds):
                dsx.resize((new,) + dsx.shape[1:])
            ps_ds[old:new] = ps
            # Maintain compatibility with the existing depth-conditioned data layout:
            # conditions[:,0] = depth [cm], conditions[:,1] = initial beam energy [MeV].
            cond_ds[old:new, 0] = np.nan if depth_cm is None else float(depth_cm)
            cond_ds[old:new, 1] = np.nan if beam_energy_MeV is None else float(beam_energy_MeV)
            w_ds[old:new] = chunk["w"].to_numpy(dtype=np.float32)
            pdg_ds[old:new] = chunk["pdg"].to_numpy(dtype=np.int32)
            dzneg_ds[old:new] = chunk["dz_neg"].to_numpy(dtype=np.int8)
            first_ds[old:new] = chunk["first"].to_numpy(dtype=np.int8)
            n_output = new
        h.attrs["schema_version"] = "1.0"
        h.attrs["source_phsp"] = str(phsp_path)
        h.attrs["source_header"] = str(paired_header(phsp_path)) if paired_header(phsp_path).exists() else ""
        h.attrs["header_json"] = json.dumps(hdr, ensure_ascii=False)
        h.attrs["header_text"] = header_text
        h.attrs["beam_energy_MeV"] = np.nan if beam_energy_MeV is None else float(beam_energy_MeV)
        h.attrs["depth_cm"] = np.nan if depth_cm is None else float(depth_cm)
        h.attrs["protons_only"] = bool(protons_only)
        h.attrs["n_input_rows"] = int(n_input)
        h.attrs["n_output_rows"] = int(n_output)
        h.attrs["n_protons_in_input"] = int(n_protons)
        h.attrs["invalid_direction_rows_clipped"] = int(n_invalid)
        h.attrs["energy_min_MeV"] = float(e_min) if np.isfinite(e_min) else np.nan
        h.attrs["energy_max_MeV"] = float(e_max) if np.isfinite(e_max) else np.nan
    return {"n_input_rows": n_input, "n_output_rows": n_output,
            "n_protons_in_input": n_protons, "invalid_direction_rows_clipped": n_invalid,
            "energy_min_MeV": float(e_min) if np.isfinite(e_min) else np.nan,
            "energy_max_MeV": float(e_max) if np.isfinite(e_max) else np.nan,
            "header_n_histories": hdr.get("n_histories"),
            "header_n_histories_reached": hdr.get("n_histories_reached"),
            "header_n_particles": hdr.get("n_particles"),
            "header_n_protons": hdr.get("species_proton")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="diagnostica uno o piu' file phsp")
    p.add_argument("files", nargs="+"); p.add_argument("--chunksize", type=int, default=250_000)
    p = sub.add_parser("convert", help="converte un singolo phsp in HDF5 mantenendo i metadati")
    p.add_argument("file"); p.add_argument("--out", required=True)
    p.add_argument("--beam-energy", type=float, default=None); p.add_argument("--depth", type=float, default=None)
    p.add_argument("--all-particles", action="store_true"); p.add_argument("--chunksize", type=int, default=250_000)
    a = ap.parse_args()
    if a.cmd == "info":
        for f in a.files:
            r = inspect_file(f, a.chunksize)
            print(f"\n=== {Path(f).name} ===")
            print(f"header: {'OK' if r['header_exists'] else 'MANCANTE'} | E0={r['beam_energy_MeV']} MeV | d={r['depth_cm']} cm | region={r['region']}")
            print(f"righe={r['n_rows_actual']:,} | protoni={r['n_protons_actual']:,} | direzioni fuori tolleranza={r['invalid_direction_rows']:,}")
            print("specie:", ", ".join(f"{k}={v:,}" for k, v in sorted(r["species_actual"].items())))
            if r["header"]: print("header stats:", json.dumps(r["header"], indent=2, ensure_ascii=False))
            for k, s in r["proton_stats"].items():
                print(f"protoni {k:>2}: mean={s['mean']:+.6g} std={s['std_population']:.6g} min={s['min']:.6g} max={s['max']:.6g}")
    else:
        inf = parse_conditions_from_name(a.file)
        e0 = a.beam_energy if a.beam_energy is not None else inf["beam_energy_MeV"]
        d = a.depth if a.depth is not None else inf["depth_cm"]
        stats = write_one_h5(a.file, a.out, e0, d, not a.all_particles, a.chunksize)
        print(f"Creato {a.out}: {json.dumps(stats, indent=2)}")
        if e0 is None or d is None:
            print("ATTENZIONE: E0 o profondita' non inferiti; conditions contiene NaN. Specificarli esplicitamente per il training.")

if __name__ == "__main__":
    main()
