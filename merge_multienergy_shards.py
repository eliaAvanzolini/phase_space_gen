#!/usr/bin/env python3
"""Optional streaming merge of per-simulation HDF5 shards into one HDF5 file.

Merged convention: phase_space=(x,y,z,dx,dy,dz,E); conditions[:,0]=depth_cm;
conditions[:,1]=initial beam energy MeV. Metadata arrays are retained.
Use only after reviewing dataset_manifest.csv. This may require substantial disk space.
"""
import argparse, csv, json
from pathlib import Path
import h5py
import numpy as np

DATASETS = {"phase_space": (np.float32, (7,)), "conditions": (np.float32, (2,)),
            "weights": (np.float32, ()), "pdg": (np.int32, ()),
            "dz_neg": (np.int8, ()), "first": (np.int8, ())}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--manifest", default=None, help="dataset_manifest.csv; if omitted, uses all *.h5 except merged output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-rows", type=int, default=100_000)
    a = ap.parse_args()
    shard_dir, out = Path(a.shard_dir).resolve(), Path(a.out).resolve()
    if a.manifest:
        with open(a.manifest, newline="", encoding="utf-8-sig") as f: rows = list(csv.DictReader(f))
        paths = [shard_dir / r["processed_file"] for r in rows if r.get("processed_file")]
    else:
        paths = sorted(p for p in shard_dir.glob("*.h5") if p.resolve() != out)
    paths = [p for p in paths if p.exists()]
    if not paths: raise SystemExit("Nessuno shard HDF5 trovato.")
    total = 0
    for p in paths:
        with h5py.File(p, "r") as h: total += len(h["phase_space"])
    print(f"Shard={len(paths)}; particelle={total:,}; output={out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as dst:
        chunk = max(1, min(a.chunk_rows, total if total else 1))
        dsets = {}
        for name, (dtype, tail) in DATASETS.items():
            dsets[name] = dst.create_dataset(name, shape=(total,) + tail, dtype=dtype,
                chunks=(chunk,) + tail, compression="gzip", compression_opts=4, shuffle=True)
        src_ids = dst.create_dataset("source_id", shape=(total,), dtype=np.int32,
            chunks=(chunk,), compression="gzip", compression_opts=4)
        source_names = []
        offset = 0
        for sid, p in enumerate(paths):
            with h5py.File(p, "r") as src:
                n = len(src["phase_space"]); source_names.append(str(p.name))
                for lo in range(0, n, a.chunk_rows):
                    hi = min(n, lo + a.chunk_rows); sl = slice(lo, hi); outsl = slice(offset + lo, offset + hi)
                    for name in DATASETS:
                        dsets[name][outsl] = src[name][sl]
                    src_ids[outsl] = sid
            offset += n
            print(f"[{sid+1}/{len(paths)}] {p.name}: {n:,} righe", flush=True)
        dst.attrs["schema_version"] = "1.0"
        dst.attrs["conditions_columns"] = "depth_cm, beam_energy_MeV"
        dst.attrs["phase_space_columns"] = "x,y,z,dx,dy,dz,E"
        dst.attrs["source_files_json"] = json.dumps(source_names)
    print("Merge completato.")

if __name__ == "__main__": main()
