"""
read_topas_phsp.py
==================
Lettura di phase space TOPAS in formato ASCII (.phsp + .header) e conversione
nel formato 7D del progetto (x, y, z, dx, dy, dz, E) / HDF5.

Colonne del file .phsp (vedi .header):
    1 x [cm]   2 y [cm]   3 z [cm]   4 dir.cos X   5 dir.cos Y   6 E [MeV]
    7 peso     8 PDG      9 flag "dz < 0" (1 = vero)   10 flag "primo scorato della history"

Nota: il terzo coseno direttore NON è salvato: va ricostruito come
    dz = +/- sqrt(1 - dx^2 - dy^2),  segno dato dalla colonna 9.

Uso da CLI:
    python read_topas_phsp.py info    Input_70MeV.phsp
    python read_topas_phsp.py convert SurfaceFile_3_5_peak_1_2_5.phsp --out surf.h5 --depth 3.5 --protons_only
    python read_topas_phsp.py plot    Input_70MeV.phsp SurfaceFile_3_5_peak_1_2_5.phsp --out plots/
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

COLS = ["x", "y", "z", "dx", "dy", "E", "w", "pdg", "dz_neg", "first"]
PDG_NAMES = {2212: "proton", 11: "e-", -11: "e+", 22: "gamma", 2112: "neutron",
             1000010020: "deuteron", 1000020040: "alpha"}
PROTON = 2212


# ─────────────────────────────── lettura ──────────────────────────────────────

def read_header(phsp_path: str | Path) -> Dict:
    """Parsa il .header gemello (stesso nome, estensione .header), se esiste."""
    hp = Path(phsp_path).with_suffix(".header")
    info: Dict = {}
    if not hp.exists():
        return info
    txt = hp.read_text()
    for key, pat in {
        "n_histories":        r"Number of Original Histories:\s*(\d+)",
        "n_histories_reached": r"Number of Original Histories that Reached Phase Space:\s*(\d+)",
        "n_particles":        r"Number of Scored Particles:\s*(\d+)",
    }.items():
        m = re.search(pat, txt)
        if m:
            info[key] = int(m.group(1))
    return info


def read_phsp(path: str | Path) -> pd.DataFrame:
    """Legge un .phsp ASCII TOPAS in un DataFrame con le 10 colonne standard."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS, engine="c",
                     dtype={"pdg": np.int64})
    df["dz_neg"] = df["dz_neg"].astype(np.int8)
    df["first"] = df["first"].astype(np.int8)
    return df


def to_7d(df: pd.DataFrame) -> np.ndarray:
    """DataFrame -> array float32 (N, 7) = (x, y, z, dx, dy, dz, E)."""
    dz2 = 1.0 - df["dx"].to_numpy(np.float64) ** 2 - df["dy"].to_numpy(np.float64) ** 2
    dz = np.sqrt(np.clip(dz2, 0.0, 1.0))
    dz = np.where(df["dz_neg"].to_numpy() == 1, -dz, dz)
    return np.column_stack([
        df["x"], df["y"], df["z"], df["dx"], df["dy"], dz, df["E"]
    ]).astype(np.float32)


# ─────────────────────────────── utilità ──────────────────────────────────────

# Z del piano di scoring a profondità 0:  z = -(Z0 + depth)  (verificato su 0.5, 3.5, 4.0 -> Z0 = 158 cm)
Z0_CM = 158.0
NAME_RE = re.compile(r"SurfaceFile_(?P<depth>\d+(?:[._]\d+)?)_(?P<region>peak|valley)_", re.IGNORECASE)


def parse_name(name: str) -> Optional[Dict]:
    """
    'SurfaceFile_3.5_peak_1_2_5.phsp' o 'SurfaceFile_3_5_peak_1_2_5.phsp' -> {'depth': 3.5, 'region': 'peak'}
    'SurfaceFile_1_valley_1_2_5.phsp' -> {'depth': 1.0, 'region': 'valley'}
    (l'upload web può trasformare il '.' in '_': entrambe le forme sono accettate)
    """
    m = NAME_RE.search(name)
    if not m:
        return None
    return {"depth": float(m.group("depth").replace("_", ".")), "region": m.group("region").lower()}


def parse_depth_from_name(name: str) -> Optional[float]:
    r = parse_name(name)
    return None if r is None else r["depth"]


def depth_from_z(z: float, z0: float = Z0_CM) -> float:
    """Profondità [cm] ricavata dalla coordinata z del piano di scoring."""
    return -float(z) - z0


def summarize(df: pd.DataFrame, header: Optional[Dict] = None, name: str = "") -> str:
    lines = [f"=== {name} : {len(df)} particelle ==="]
    if header:
        lines.append(f"header: {header}")
    vc = df["pdg"].value_counts()
    lines.append("specie: " + ", ".join(f"{PDG_NAMES.get(k, k)}={v}" for k, v in vc.items()))
    p = df[df["pdg"] == PROTON]
    lines.append(f"z unici: {np.unique(df['z'])[:5]}")
    for c in ["x", "y", "dx", "dy", "E"]:
        lines.append(f"  protoni {c:>2}: mean={p[c].mean():+.4f} std={p[c].std():.4f} "
                     f"min={p[c].min():+.4f} max={p[c].max():+.4f}")
    lines.append(f"frazione dz<0: {df['dz_neg'].mean():.4f}   pesi unici: {np.unique(df['w'])[:5]}")
    return "\n".join(lines)


def save_h5(path: str | Path, ps7: np.ndarray, cond: Optional[np.ndarray] = None) -> None:
    """Salva nello stesso layout usato da data/dataset.py (phase_space, conditions)."""
    import h5py
    with h5py.File(path, "w") as f:
        f.create_dataset("phase_space", data=ps7, compression="gzip")
        if cond is not None:
            f.create_dataset("conditions", data=cond, compression="gzip")


def load_filtered(path: str | Path, protons_only: bool = True,
                  min_E: float = 0.0) -> Tuple[np.ndarray, Dict]:
    df = read_phsp(path)
    hdr = read_header(path)
    if protons_only:
        df = df[df["pdg"] == PROTON]
    if min_E > 0:
        df = df[df["E"] > min_E]
    return to_7d(df), hdr


# ─────────────────────────────── plotting ─────────────────────────────────────

def quick_plots(files, out_dir: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {Path(f).stem: read_phsp(f) for f in files}
    data = {k: v[v["pdg"] == PROTON] for k, v in data.items()}

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for name, d in data.items():
        for a, c in zip(ax.ravel(), ["x", "y", "dx", "dy", "E"]):
            a.hist(d[c], bins=100, histtype="step", density=True, label=name)
            a.set_xlabel(c)
        ax[1, 2].axis("off")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("Protoni: marginali (normalizzate a densità)")
    fig.tight_layout()
    fig.savefig(out_dir / "marginals.png", dpi=110)

    fig, ax = plt.subplots(1, len(data), figsize=(6 * len(data), 5), squeeze=False)
    for a, (name, d) in zip(ax[0], data.items()):
        a.hist2d(d["y"], d["dy"], bins=80)
        a.set_xlabel("y [cm]"); a.set_ylabel("dy"); a.set_title(name, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "y_dy.png", dpi=110)
    print(f"plot salvati in {out_dir}")


# ─────────────────────────────── CLI ──────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info");    p.add_argument("files", nargs="+")
    p = sub.add_parser("plot");    p.add_argument("files", nargs="+"); p.add_argument("--out", default="plots")
    p = sub.add_parser("convert")
    p.add_argument("file"); p.add_argument("--out", required=True)
    p.add_argument("--depth", type=float, default=None, help="profondità [cm] -> colonna 'conditions'")
    p.add_argument("--protons_only", action="store_true")
    a = ap.parse_args()

    if a.cmd == "info":
        for f in a.files:
            print(summarize(read_phsp(f), read_header(f), Path(f).name), "\n")
    elif a.cmd == "plot":
        quick_plots(a.files, a.out)
    elif a.cmd == "convert":
        ps, _ = load_filtered(a.file, a.protons_only)
        d = a.depth if a.depth is not None else parse_depth_from_name(Path(a.file).name)
        cond = None if d is None else np.full((len(ps), 1), d, dtype=np.float32)
        save_h5(a.out, ps, cond)
        print(f"salvato {a.out}: ps {ps.shape}, depth={d}")


if __name__ == "__main__":
    main()

def write_topas_ascii(ps7: np.ndarray, path: str | Path) -> None:
    """Salva un array 7D (x,y,z,dx,dy,dz,E) in formato ASCII TOPAS a 10 colonne."""
    N = len(ps7)
    arr = np.column_stack([
        ps7[:, 0], ps7[:, 1], ps7[:, 2],
        ps7[:, 3], ps7[:, 4], ps7[:, 6],
        np.ones(N, dtype=np.float32),
        np.full(N, 2212, dtype=np.int64),
        (ps7[:, 5] < 0).astype(np.int8),
        np.ones(N, dtype=np.int8)
    ])
    fmt = ["%12.6f", "%12.6f", "%12.6f", "%12.6f", "%12.6f", "%12.6f", "%d", "%d", "%d", "%d"]
    np.savetxt(path, arr, fmt=fmt)
