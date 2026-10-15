#!/usr/bin/env python3
"""Discover paired TOPAS .phsp/.header files and build HDF5 shards plus manifests.

Recommended workflow:
  1) scan -> creates a manifest template, inferred only from filenames.
  2) inspect/edit the manifest to fill missing E0/depth and include flags.
  3) build -> converts each selected simulation to one HDF5 shard.

The script does not assume that an Excel column called File maps to .phsp files.
It uses only actual .phsp filenames and their same-stem .header companions.
"""
from __future__ import annotations
import argparse, csv, json, re, sys
from pathlib import Path
from typing import Dict, List

from read_topas_phsp_multienergy import (paired_header, parse_conditions_from_name,
                                         read_header, write_one_h5)

FIELDS = ["phsp_file", "header_file", "beam_energy_MeV", "depth_cm", "region", "include", "geometry_id", "notes"]

def discover(root: Path) -> List[Dict[str, str]]:
    rows = []
    for phsp in sorted(root.rglob("*.phsp")):
        hdr = paired_header(phsp)
        c = parse_conditions_from_name(phsp.name)
        parsed = c["beam_energy_MeV"] is not None and c["depth_cm"] is not None
        rows.append({
            "phsp_file": str(phsp.relative_to(root)),
            "header_file": str(hdr.relative_to(root)) if hdr.exists() else "",
            "beam_energy_MeV": "" if c["beam_energy_MeV"] is None else f"{c['beam_energy_MeV']:g}",
            "depth_cm": "" if c["depth_cm"] is None else f"{c['depth_cm']:g}",
            "region": c["region"] or "",
            # Safe-by-default: require an explicit human review before conversion.
            "include": "0",
            "geometry_id": "",
            "notes": ("REVIEW E0/depth/header/geometry, then set include=1" if parsed and hdr.exists()
                      else "VERIFY: condition not inferred and/or header missing; fill fields before include=1"),
        })
    return rows

def write_csv(path: Path, rows: List[Dict[str, str]], fields: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def is_true(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y", "si", "sì"}

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_.") or "simulation"

def scan_cmd(a):
    root = Path(a.root).resolve()
    if not root.is_dir(): raise SystemExit(f"Directory non trovata: {root}")
    rows = discover(root)
    out = Path(a.manifest)
    write_csv(out, rows, FIELDS)
    print(f"Trovati {len(rows)} file .phsp sotto {root}")
    print(f"Manifest template: {out.resolve()}")
    missing_header = sum(not r["header_file"] for r in rows)
    missing_conditions = sum(not r["beam_energy_MeV"] or not r["depth_cm"] for r in rows)
    included = sum(is_true(r["include"]) for r in rows)
    print(f"Coppie con header presente: {len(rows)-missing_header}/{len(rows)}")
    print(f"Condizioni E0+d inferite automaticamente: {len(rows)-missing_conditions}/{len(rows)}")
    print(f"Righe marcate per il build: {included}/{len(rows)} (impostazione sicura: tutte escluse fino a revisione)")
    if rows:
        print("Apri il CSV, verifica energia iniziale, profondita', header e geometria; poi imposta include=1 SOLO sulle righe verificate.")
    if missing_header or missing_conditions:
        print("ATTENZIONE: ci sono righe con header o condizioni mancanti; completarle manualmente prima di includerle.")
    print("Non assumere che una condizione inferita dal nome sia corretta: controllarla contro la documentazione della simulazione.")

def build_cmd(a):
    root = Path(a.root).resolve(); manifest = Path(a.manifest).resolve(); outdir = Path(a.out_dir).resolve()
    if not root.is_dir(): raise SystemExit(f"Directory non trovata: {root}")
    if not manifest.exists(): raise SystemExit(f"Manifest non trovato: {manifest}. Esegui prima il comando scan.")
    rows = read_csv(manifest); outdir.mkdir(parents=True, exist_ok=True)
    reports = []; seen_outputs = set(); errors = 0
    for idx, r in enumerate(rows, start=1):
        if not is_true(r.get("include", "0")): continue
        rel = Path(r["phsp_file"])
        phsp = (root / rel).resolve()
        try: phsp.relative_to(root)
        except ValueError:
            print(f"[SKIP] path fuori root: {phsp}"); errors += 1; continue
        hdr = paired_header(phsp)
        if not phsp.exists():
            print(f"[SKIP] phsp mancante: {phsp}"); errors += 1; continue
        if not hdr.exists():
            print(f"[SKIP] header gemello mancante: {hdr}"); errors += 1; continue
        try:
            e0 = float(r["beam_energy_MeV"]); depth = float(r["depth_cm"])
        except (ValueError, TypeError):
            print(f"[SKIP] E0/depth non numerici nel manifest: {phsp.name}"); errors += 1; continue
        outname = safe_name(phsp.stem) + ".h5"
        if outname in seen_outputs:
            outname = safe_name(f"{idx:03d}_{phsp.stem}") + ".h5"
        seen_outputs.add(outname)
        outpath = outdir / outname
        print(f"[{idx}/{len(rows)}] {phsp.name}: E0={e0:g} MeV, d={depth:g} cm -> {outname}", flush=True)
        stats = write_one_h5(phsp, outpath, e0, depth, protons_only=not a.all_particles, chunksize=a.chunksize)
        parsed_hdr, _ = read_header(phsp)
        warning = ""
        if stats.get("header_n_particles") is not None and stats["header_n_particles"] != stats["n_input_rows"]:
            warning = f"header n_particles={stats['header_n_particles']} != righe lette={stats['n_input_rows']}"
            print("  WARNING:", warning)
        if stats.get("header_n_protons") is not None and stats["header_n_protons"] != stats["n_protons_in_input"]:
            warning += ("; " if warning else "") + f"header protons={stats['header_n_protons']} != protons letti={stats['n_protons_in_input']}"
            print("  WARNING:", warning)
        reports.append({
            **r, "processed_file": str(outpath.relative_to(outdir)),
            **stats, "geometry_id": r.get("geometry_id", ""), "validation_warning": warning,
            "protons_only": str(not a.all_particles),
        })
    fields = list(dict.fromkeys(FIELDS + [k for row in reports for k in row.keys()]))
    write_csv(outdir / "dataset_manifest.csv", reports, fields)
    print(f"\nShard creati: {len(reports)}; righe non processate/problematiche: {errors}")
    print(f"Manifest dataset: {outdir / 'dataset_manifest.csv'}")
    if len(reports) == 0:
        print("ATTENZIONE: nessuno shard creato; controlla che almeno una riga verificata abbia include=1.")
    if errors:
        print("Alcuni input sono stati saltati: leggi i messaggi sopra e correggi il manifest.")
    else:
        print("Conversione completata. Verificare gli avvisi e le condizioni prima del training.")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan", help="scansiona i phsp e crea un template CSV")
    p.add_argument("--root", required=True); p.add_argument("--manifest", required=True)
    p = sub.add_parser("build", help="crea un HDF5 per simulazione, più manifest")
    p.add_argument("--root", required=True); p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", required=True); p.add_argument("--chunksize", type=int, default=250_000)
    p.add_argument("--all-particles", action="store_true", help="non filtrare i protoni (default: solo protoni)")
    a = ap.parse_args()
    if a.cmd == "scan": scan_cmd(a)
    else: build_cmd(a)

if __name__ == "__main__": main()
