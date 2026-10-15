#!/usr/bin/env python3
"""Streaming quality checks for the multi-energy TOPAS HDF5 shards.

Checks manifest/HDF5 consistency, conditions and z-depth relation, finite values,
proton filtering, direction cosines, weights, energy ranges, and basic summaries.
Does not modify any input files.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import h5py


def fnum(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else np.nan
    except (TypeError, ValueError):
        return np.nan


def approx_quantiles(sample, qs):
    if not sample:
        return {f"E_q{q}": np.nan for q in qs}
    vv = np.percentile(np.asarray(sample, dtype=np.float64), qs)
    return {f"E_q{q}": float(v) for q, v in zip(qs, vv)}


def inspect_shard(path: Path, row: dict, max_quantile_sample: int, chunksize: int) -> dict:
    base = {
        "phsp_file": row.get("phsp_file", ""),
        "processed_file": row.get("processed_file", path.name),
        "beam_energy_MeV": fnum(row.get("beam_energy_MeV")),
        "depth_cm": fnum(row.get("depth_cm")),
        "manifest_n_input_rows": fnum(row.get("n_input_rows")),
        "manifest_n_output_rows": fnum(row.get("n_output_rows")),
        "header_n_particles": fnum(row.get("header_n_particles")),
        "header_n_protons": fnum(row.get("header_n_protons")),
        "h5_path": str(path),
        "file_exists": path.exists(),
    }
    if not path.exists():
        base["qa_status"] = "MISSING_H5"
        base["qa_notes"] = "HDF5 non trovato"
        return base

    notes = []
    expected_e0, expected_d = base["beam_energy_MeV"], base["depth_cm"]
    with h5py.File(path, "r") as h:
        required = ["phase_space", "conditions", "weights", "pdg", "dz_neg", "first"]
        missing = [k for k in required if k not in h]
        if missing:
            base["qa_status"] = "MISSING_DATASETS"
            base["qa_notes"] = "Dataset mancanti: " + ",".join(missing)
            return base

        ps, cond = h["phase_space"], h["conditions"]
        n = int(ps.shape[0])
        base.update({
            "h5_n_rows": n,
            "phase_space_ndim": int(ps.ndim),
            "phase_space_ncols": int(ps.shape[1]) if ps.ndim == 2 else np.nan,
            "depth_attr_cm": fnum(h.attrs.get("depth_cm")),
            "E0_attr_MeV": fnum(h.attrs.get("beam_energy_MeV")),
            "protons_only_attr": h.attrs.get("protons_only", None),
            "conditions_depth_min": np.nan,
            "conditions_depth_max": np.nan,
            "conditions_E0_min": np.nan,
            "conditions_E0_max": np.nan,
            "n_conditions_mismatch": 0,
            "n_nonfinite_phase_space_rows": 0,
            "n_nonfinite_conditions_rows": 0,
            "n_nonfinite_weights": 0,
            "n_invalid_dz_neg_flags": 0,
            "n_invalid_first_flags": 0,
            "n_nonproton_pdg": 0,
            "n_negative_energy": 0,
            "n_zero_energy_le_1e-8": 0,
            "n_energy_gt_E0_plus_0p5": 0,
            "n_direction_xy_outside_unit_disk": 0,
            "max_abs_direction_norm_minus_1": np.nan,
            "max_abs_z_plus_depth": np.nan,
            "n_rows_z_depth_mismatch_gt_0p01": 0,
            "weight_min": np.nan,
            "weight_max": np.nan,
            "weight_mean": np.nan,
            "frac_dz_negative": np.nan,
            "frac_first_flag": np.nan,
        })

        if ps.ndim != 2 or ps.shape[1] != 7:
            notes.append(f"phase_space shape inattesa {ps.shape}")
        if cond.ndim != 2 or cond.shape[0] != n or cond.shape[1] < 2:
            notes.append(f"conditions shape inattesa {cond.shape}")
        for key, expected, actual in [
            ("depth", expected_d, base["depth_attr_cm"]),
            ("E0", expected_e0, base["E0_attr_MeV"]),
        ]:
            if np.isfinite(expected) and np.isfinite(actual) and abs(expected - actual) > 1e-5:
                notes.append(f"{key} manifest={expected:g} != attr={actual:g}")

        # Per-shard deterministic sample for approximate energy quantiles.
        stride = max(1, int(np.ceil(max(n, 1) / max(max_quantile_sample, 1))))
        E_sample = []
        count = 0
        sums_e = 0.0
        sums_e2 = 0.0
        e_min, e_max = np.inf, -np.inf
        w_min, w_max, w_sum, w_n = np.inf, -np.inf, 0.0, 0
        dz_neg_sum = first_sum = 0
        max_norm_err = 0.0
        max_z_err = 0.0
        z_mismatch_count = 0

        for start in range(0, n, chunksize):
            end = min(start + chunksize, n)
            arr = np.asarray(ps[start:end], dtype=np.float64)
            c = np.asarray(cond[start:end], dtype=np.float64)
            w = np.asarray(h["weights"][start:end], dtype=np.float64)
            pdg = np.asarray(h["pdg"][start:end])
            dzneg = np.asarray(h["dz_neg"][start:end]).astype(bool)
            first = np.asarray(h["first"][start:end]).astype(bool)

            if arr.size:
                bad_rows = ~np.isfinite(arr).all(axis=1)
                base["n_nonfinite_phase_space_rows"] += int(bad_rows.sum())
                x, y, z, dx, dy, dz, E = (arr[:, i] for i in range(7))
                base["n_negative_energy"] += int((E < -1e-7).sum())
                base["n_zero_energy_le_1e-8"] += int((E <= 1e-8).sum())
                if np.isfinite(expected_e0):
                    base["n_energy_gt_E0_plus_0p5"] += int((E > expected_e0 + 0.5).sum())
                disk = dx * dx + dy * dy
                base["n_direction_xy_outside_unit_disk"] += int((disk > 1.0 + 1e-6).sum())
                nerr = np.abs(np.sqrt(dx * dx + dy * dy + dz * dz) - 1.0)
                if len(nerr): max_norm_err = max(max_norm_err, float(np.nanmax(nerr)))
                if np.isfinite(expected_d):
                    zerr = np.abs(z + expected_d)
                    if len(zerr): max_z_err = max(max_z_err, float(np.nanmax(zerr)))
                    z_mismatch_count += int((zerr > 0.01).sum())
                good_e = E[np.isfinite(E)]
                if len(good_e):
                    count += len(good_e)
                    sums_e += float(good_e.sum())
                    sums_e2 += float(np.square(good_e).sum())
                    e_min = min(e_min, float(good_e.min()))
                    e_max = max(e_max, float(good_e.max()))
                    local_indices = np.arange(start, start + len(E))
                    take = (local_indices % stride) == 0
                    if np.any(take): E_sample.extend(E[take & np.isfinite(E)].tolist())
                base["n_nonproton_pdg"] += int((pdg != 2212).sum())
                dz_neg_sum += int(dzneg.sum())
                first_sum += int(first.sum())
                raw_dzneg = np.asarray(h["dz_neg"][start:end])
                raw_first = np.asarray(h["first"][start:end])
                base["n_invalid_dz_neg_flags"] += int((~np.isin(raw_dzneg, [0, 1])).sum())
                base["n_invalid_first_flags"] += int((~np.isin(raw_first, [0, 1])).sum())

            if c.size:
                badc = ~np.isfinite(c[:, :2]).all(axis=1)
                base["n_nonfinite_conditions_rows"] += int(badc.sum())
                if np.isfinite(expected_d):
                    base["n_conditions_mismatch"] += int((np.abs(c[:, 0] - expected_d) > 1e-5).sum())
                if np.isfinite(expected_e0):
                    base["n_conditions_mismatch"] += int((np.abs(c[:, 1] - expected_e0) > 1e-4).sum())
                if len(c):
                    cdmin, cdmax = float(np.nanmin(c[:, 0])), float(np.nanmax(c[:, 0]))
                    cemin, cemax = float(np.nanmin(c[:, 1])), float(np.nanmax(c[:, 1]))
                    base["conditions_depth_min"] = cdmin if not np.isfinite(base["conditions_depth_min"]) else min(base["conditions_depth_min"], cdmin)
                    base["conditions_depth_max"] = cdmax if not np.isfinite(base["conditions_depth_max"]) else max(base["conditions_depth_max"], cdmax)
                    base["conditions_E0_min"] = cemin if not np.isfinite(base["conditions_E0_min"]) else min(base["conditions_E0_min"], cemin)
                    base["conditions_E0_max"] = cemax if not np.isfinite(base["conditions_E0_max"]) else max(base["conditions_E0_max"], cemax)

            if len(w):
                base["n_nonfinite_weights"] += int((~np.isfinite(w)).sum())
                goodw = w[np.isfinite(w)]
                if len(goodw):
                    w_min = min(w_min, float(goodw.min()))
                    w_max = max(w_max, float(goodw.max()))
                    w_sum += float(goodw.sum())
                    w_n += len(goodw)

        mean_e = sums_e / count if count else np.nan
        std_e = np.sqrt(max(sums_e2 / count - mean_e**2, 0.0)) if count else np.nan
        base.update({
            "energy_n_finite": count,
            "energy_mean_MeV": mean_e,
            "energy_std_MeV": std_e,
            "energy_min_MeV_actual": e_min if np.isfinite(e_min) else np.nan,
            "energy_max_MeV_actual": e_max if np.isfinite(e_max) else np.nan,
            "weight_min": w_min if np.isfinite(w_min) else np.nan,
            "weight_max": w_max if np.isfinite(w_max) else np.nan,
            "weight_mean": w_sum / w_n if w_n else np.nan,
            "frac_dz_negative": dz_neg_sum / n if n else np.nan,
            "frac_first_flag": first_sum / n if n else np.nan,
            "max_abs_direction_norm_minus_1": max_norm_err,
            "max_abs_z_plus_depth": max_z_err,
            "n_rows_z_depth_mismatch_gt_0p01": z_mismatch_count,
        })
        base.update(approx_quantiles(E_sample, [1, 5, 25, 50, 75, 95, 99]))

        exp_out = fnum(row.get("n_output_rows"))
        if np.isfinite(exp_out) and exp_out != n:
            notes.append(f"righe H5={n} != manifest n_output_rows={exp_out:g}")
        exp_in = fnum(row.get("n_input_rows"))
        hnp = fnum(row.get("header_n_particles"))
        if np.isfinite(exp_in) and np.isfinite(hnp) and exp_in != hnp:
            notes.append(f"n_input_rows={exp_in:g} != header_n_particles={hnp:g}")
        hp = fnum(row.get("header_n_protons"))
        nprot = fnum(row.get("n_protons_in_input"))
        if np.isfinite(hp) and np.isfinite(nprot) and hp != nprot:
            notes.append(f"header_n_protons={hp:g} != n_protons_in_input={nprot:g}")
        if base["n_conditions_mismatch"]:
            notes.append(f"{base['n_conditions_mismatch']} valori conditions diversi da E0/depth manifest")
        if (base["n_nonfinite_phase_space_rows"] or base["n_nonfinite_conditions_rows"]
                or base["n_nonfinite_weights"]):
            notes.append("valori non finiti")
        if base["n_invalid_dz_neg_flags"] or base["n_invalid_first_flags"]:
            notes.append("flag TOPAS non binari (attesi 0/1)")
        if base["n_nonproton_pdg"]:
            notes.append("HDF5 protons-only contiene PDG diversi da 2212")
        if base["n_negative_energy"]:
            notes.append("energie negative")
        if base["n_direction_xy_outside_unit_disk"]:
            notes.append("dx^2+dy^2 > 1 oltre tolleranza")
        if max_norm_err > 2e-5:
            notes.append(f"direzione non unitaria max_err={max_norm_err:.3g}")
        if z_mismatch_count:
            notes.append("z non coerente con z=-depth secondo convenzione attesa; verificare geometria")
        if np.isfinite(expected_e0) and base["n_energy_gt_E0_plus_0p5"]:
            notes.append("alcune energie > E0+0.5 MeV: controllare secondari/definizione E0")
        if np.isfinite(w_min) and (w_min < 0 or not np.isfinite(w_min)):
            notes.append("pesi non positivi/non validi")

    base["qa_notes"] = "; ".join(notes)
    base["qa_status"] = "REVIEW" if notes else "OK"
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/processed_multienergy/dataset_manifest.csv")
    ap.add_argument("--h5-dir", default="data/processed_multienergy")
    ap.add_argument("--out", default="data/metadata/qa_multienergy_report.csv")
    ap.add_argument("--summary-out", default="data/metadata/qa_multienergy_by_E0.csv")
    ap.add_argument("--chunksize", type=int, default=250_000)
    ap.add_argument("--max-quantile-sample", type=int, default=50_000)
    ap.add_argument("--only-included", action="store_true", help="considera solo include=1 se la colonna esiste")
    a = ap.parse_args()

    manifest_path = Path(a.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest non trovato: {manifest_path}")
    df = pd.read_csv(manifest_path)
    if a.only_included and "include" in df:
        df = df[df["include"].astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y"])].copy()

    root = Path(a.h5_dir)
    results = []
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        rel = str(r.get("processed_file", ""))
        path = Path(rel)
        if not path.is_absolute():
            path = root / path
        print(f"[{i}/{len(df)}] QA {path.name}", flush=True)
        results.append(inspect_shard(path, r.to_dict(), a.max_quantile_sample, a.chunksize))

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(results)
    report.to_csv(out, index=False)
    print(f"\nReport completo: {out}")
    if "qa_status" in report:
        print("Status:\n", report["qa_status"].value_counts(dropna=False).to_string())
    if "qa_notes" in report:
        bad = report[report["qa_notes"].fillna("").str.len() > 0]
        print(f"Shard con note da esaminare: {len(bad)}/{len(report)}")
        if len(bad):
            print(bad[["phsp_file", "beam_energy_MeV", "depth_cm", "qa_notes"]].to_string(index=False, max_colwidth=150))

    good = report.dropna(subset=["beam_energy_MeV"]).groupby("beam_energy_MeV").agg(
        n_planes=("phsp_file", "count"),
        n_protons=("h5_n_rows", "sum"),
        min_depth_cm=("depth_cm", "min"),
        max_depth_cm=("depth_cm", "max"),
        n_review=("qa_status", lambda x: int((x != "OK").sum())),
    ).reset_index()
    sout = Path(a.summary_out); sout.parent.mkdir(parents=True, exist_ok=True)
    good.to_csv(sout, index=False)
    print(f"Riepilogo per energia: {sout}")
    print(good.to_string(index=False))

if __name__ == "__main__":
    main()
