"""
data/read_iaea_phsp.py
=======================
Legge i file di phase space IAEA (.IAEAphsp) e li converte in HDF5.

Supporta AUTOMATICAMENTE qualsiasi file IAEA:
    - Parsing dell'header (.IAEAheader) per rilevare record_size, Z costante,
      contenuti del record (X, Y, Z, U, V, W stored?) e byte order
    - 6MV e 10MV (testato su Elekta Precise IAEA 300-305)
    - Multi-file: fornire più --input per concatenare automaticamente

Uso:
    # Singolo file 6MV
    python data/read_iaea_phsp.py \\
        --input data/ELEKTA_PRECISE_6mv_part1_big.IAEAphsp \\
        --output data/elekta_6mv.h5 --stats

    # Multi-file 10MV (part1 + part2)
    python data/read_iaea_phsp.py \\
        --input data/ELEKTA_PRECISE_10mv_part1.IAEAphsp \\
               data/ELEKTA_PRECISE_10mv_part2.IAEAphsp \\
        --output data/elekta_10mv.h5 --stats

    # Con label di energia (per dataset condizionale)
    python data/read_iaea_phsp.py \\
        --input data/ELEKTA_PRECISE_6mv_part1_big.IAEAphsp \\
        --output data/elekta_6mv_labeled.h5 \\
        --E_nom 6.0 --jaw_x 5.0 --jaw_y 5.0
"""

import sys
import re
import struct
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Parsing automatico dell'header IAEA ──────────────────────────────────────

def parse_iaea_header(header_path: str) -> dict:
    """
    Parsa un file .IAEAheader e restituisce un dizionario con:
        record_length : int (byte per record)
        z_constant    : float (Z costante in cm, da $RECORD_CONSTANT)
        n_particles   : int (numero totale di particelle)
        n_photons     : int
        contents      : dict con flag X, Y, Z, U, V, W stored, weight, extra_floats, extra_longs
        e_max         : float (energia massima dei fotoni)
        title         : str
    """
    header_path = Path(header_path)
    if not header_path.exists():
        raise FileNotFoundError(f"Header non trovato: {header_path}")

    text = header_path.read_text(encoding="utf-8", errors="replace")

    info = {}

    # Record length
    m = re.search(r'\$RECORD_LENGTH:\s*\n\s*(\d+)', text)
    info["record_length"] = int(m.group(1)) if m else 33

    # Z constant
    m = re.search(r'\$RECORD_CONSTANT:\s*\n\s*([-\d.]+)', text)
    info["z_constant"] = float(m.group(1)) if m else 0.0

    # Particles / Photons
    m = re.search(r'\$PARTICLES:\s*\n\s*(\d+)', text)
    info["n_particles"] = int(m.group(1)) if m else 0

    m = re.search(r'\$PHOTONS:\s*\n\s*(\d+)', text)
    info["n_photons"] = int(m.group(1)) if m else 0

    # Title
    m = re.search(r'\$TITLE:\s*\n\s*(.+?)(?:\n|$)', text)
    info["title"] = m.group(1).strip() if m else ""

    # Record contents (X, Y, Z, U, V, W, Weight, extra_floats, extra_longs)
    m = re.search(r'\$RECORD_CONTENTS:\s*\n((?:\s+\d+.*\n)+)', text)
    if m:
        lines = m.group(1).strip().split('\n')
        vals = []
        for line in lines:
            v = re.match(r'\s*(\d+)', line)
            if v:
                vals.append(int(v.group(1)))
        # Ordine da standard IAEA: X, Y, Z, U, V, W, Weight, extra_floats, extra_longs
        contents = {
            "X": vals[0] if len(vals) > 0 else 1,
            "Y": vals[1] if len(vals) > 1 else 1,
            "Z": vals[2] if len(vals) > 2 else 0,
            "U": vals[3] if len(vals) > 3 else 1,
            "V": vals[4] if len(vals) > 4 else 1,
            "W": vals[5] if len(vals) > 5 else 1,
            "weight": vals[6] if len(vals) > 6 else 1,
            "extra_floats": vals[7] if len(vals) > 7 else 0,
            "extra_longs": vals[8] if len(vals) > 8 else 0,
        }
    else:
        contents = {
            "X": 1, "Y": 1, "Z": 0, "U": 1, "V": 1, "W": 1,
            "weight": 1, "extra_floats": 0, "extra_longs": 0,
        }
    info["contents"] = contents

    # Emax dei fotoni
    m = re.search(r'PHOTONS\s*$', text, re.MULTILINE)
    if m:
        # Cerca la riga con le statistiche dei fotoni
        stats_match = re.search(
            r'([\d.Ee+\-]+)\s+([\d.Ee+\-]+)\s+([\d.Ee+\-]+)\s+([\d.Ee+\-]+)\s+'
            r'([\d.Ee+\-]+)\s+([\d.Ee+\-]+)\s+PHOTONS',
            text
        )
        if stats_match:
            info["e_max"] = float(stats_match.group(6))
            info["e_mean"] = float(stats_match.group(4))
        else:
            info["e_max"] = 20.0
            info["e_mean"] = 0.0
    else:
        info["e_max"] = 20.0
        info["e_mean"] = 0.0

    # Byte order
    m = re.search(r'\$BYTE_ORDER:\s*\n\s*(\d+)', text)
    info["byte_order"] = m.group(1).strip() if m else "1234"

    return info


def _find_header(phsp_path: Path) -> Path:
    """
    Dato un file .IAEAphsp, trova il corrispondente .IAEAheader.
    Strategia: stessa base name ma con estensione .IAEAheader.
    """
    # Prova con lo stesso nome esatto
    header = phsp_path.with_suffix(".IAEAheader")
    if header.exists():
        return header

    # Prova con il nome base (senza _big, _part1, ecc.)
    name = phsp_path.stem
    for suffix in ["_big", "_part1", "_part2", " (1)", " (2)"]:
        name = name.replace(suffix, "")

    candidates = list(phsp_path.parent.glob(f"*{name}*.IAEAheader"))
    # Preferisci il part corrispondente
    for c in candidates:
        if phsp_path.stem.replace("_big", "") in c.stem:
            return c
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Header IAEA non trovato per {phsp_path}. "
        f"Cerco {header} o pattern *{name}*.IAEAheader in {phsp_path.parent}"
    )


# ─── Lettura del file binario IAEA ────────────────────────────────────────────

def read_iaea_phsp(
    phsp_path: str,
    max_n: int = None,
    remove_511: bool = True,
    header_info: dict = None,
    spatial_filter_cm: float = None,
) -> np.ndarray:
    """
    Legge un file .IAEAphsp e restituisce un array (N, 7) float32.
    Colonne: [x_cm, y_cm, z_cm, dx, dy, dz, E_MeV]

    Il record_size e la Z costante vengono letti AUTOMATICAMENTE dall'header.

    Parameters
    ----------
    phsp_path : path al file .IAEAphsp
    max_n     : numero massimo di record da leggere
    remove_511: rimuovi il picco di annichilazione a 511 keV
    header_info : dizionario precompilato (se None, viene parsato dall'header)
    spatial_filter_cm : se != None, filtra |x| e |y| < spatial_filter_cm
    """
    phsp_path = Path(phsp_path)

    # Parsa l'header se non fornito
    if header_info is None:
        header_path = _find_header(phsp_path)
        header_info = parse_iaea_header(str(header_path))
        print(f"  Header: {header_path.name}")
        print(f"    Title:        {header_info['title']}")
        print(f"    Record size:  {header_info['record_length']} byte")
        print(f"    Z constant:   {header_info['z_constant']} cm")
        print(f"    Particles:    {header_info['n_particles']:,}")
        print(f"    Photons:      {header_info['n_photons']:,}")
        print(f"    E_max:        {header_info['e_max']:.3f} MeV")

    record_size = header_info["record_length"]
    z_const = header_info["z_constant"]
    contents = header_info["contents"]

    file_size = phsp_path.stat().st_size
    n_records = file_size // record_size
    n_to_read = min(n_records, max_n) if max_n else n_records

    print(f"\n  File: {phsp_path.name} ({file_size/1e9:.2f} GB)")
    print(f"  Records nel file (da {record_size} byte): {n_records:,}")
    print(f"  Da leggere:        {n_to_read:,}")

    # ── Calcola gli offset dei campi nel record ──────────────────────────────
    # Layout standard IAEA: [particle_type(1)] [E(4)] [X(4)] [Y(4)] [Z(4)?]
    #                        [U(4)] [V(4)] [W(4)?] [weight(4)?]
    #                        [extra_floats(4*N)] [extra_longs(4*N)]
    offset = 1  # primo byte: particle type (int8)
    off_E = offset; offset += 4
    off_X = offset; offset += 4 if contents["X"] else 0
    off_Y = offset; offset += 4 if contents["Y"] else 0
    off_Z = offset; offset += 4 if contents["Z"] else 0
    off_U = offset; offset += 4 if contents["U"] else 0
    off_V = offset; offset += 4 if contents["V"] else 0
    off_W = offset; offset += 4 if contents["W"] else 0

    chunk_size = 500_000
    results = []
    n_read = 0
    n_kept = 0

    with open(phsp_path, "rb") as f:
        while n_read < n_to_read:
            n_chunk = min(chunk_size, n_to_read - n_read)
            raw = f.read(n_chunk * record_size)
            if not raw:
                break

            actual = len(raw) // record_size
            if actual == 0:
                break

            buf = np.frombuffer(raw[:actual * record_size], dtype=np.uint8)
            buf = buf.reshape(actual, record_size)

            # Energia (byte 1-5)
            E_raw = np.frombuffer(buf[:, off_E:off_E+4].tobytes(), dtype=np.float32)
            E_abs = np.abs(E_raw)

            # Posizione X, Y
            X = np.frombuffer(buf[:, off_X:off_X+4].tobytes(), dtype=np.float32) if contents["X"] else np.zeros(actual, dtype=np.float32)
            Y = np.frombuffer(buf[:, off_Y:off_Y+4].tobytes(), dtype=np.float32) if contents["Y"] else np.zeros(actual, dtype=np.float32)

            # Posizione Z (costante o letta)
            if contents["Z"]:
                Z = np.frombuffer(buf[:, off_Z:off_Z+4].tobytes(), dtype=np.float32)
            else:
                Z = np.full(actual, z_const, dtype=np.float32)

            # Direzione U, V (dx, dy)
            u = np.frombuffer(buf[:, off_U:off_U+4].tobytes(), dtype=np.float32) if contents["U"] else np.zeros(actual, dtype=np.float32)
            v = np.frombuffer(buf[:, off_V:off_V+4].tobytes(), dtype=np.float32) if contents["V"] else np.zeros(actual, dtype=np.float32)

            # Direzione W (dz): letta o calcolata
            if contents["W"]:
                w = np.frombuffer(buf[:, off_W:off_W+4].tobytes(), dtype=np.float32)
            else:
                w2 = 1.0 - (u**2 + v**2)
                w = np.sqrt(np.maximum(0.0, w2))

            # Filtri fisici
            valid = (E_abs > 0.001) & (E_abs < 20.0)

            if remove_511:
                valid &= ~((E_abs > 0.505) & (E_abs < 0.520))

            if spatial_filter_cm is not None:
                valid &= (np.abs(X) <= spatial_filter_cm) & (np.abs(Y) <= spatial_filter_cm)

            ps = np.column_stack([
                X[valid], Y[valid], Z[valid],
                u[valid], v[valid], w[valid],
                E_abs[valid]
            ]).astype(np.float32)

            results.append(ps)
            n_read += actual
            n_kept += ps.shape[0]

            if n_read % 1_000_000 == 0 or n_read >= n_to_read:
                print(f"\r  Letti {n_read:>10,} / {n_to_read:,}  "
                      f"(kept {n_kept:,}, {100*n_kept/max(n_read,1):.1f}%)", end="")

    print()

    if not results:
        raise ValueError("Nessun dato valido letto dal file phase space.")

    ps_all = np.concatenate(results, axis=0)

    # Forza normalizzazione del vettore direzione (||d||=1)
    d_norms = np.linalg.norm(ps_all[:, 3:6], axis=1, keepdims=True)
    ps_all[:, 3:6] /= np.where(d_norms > 0, d_norms, 1.0)

    print(f"\n  Totale particelle mantenute: {len(ps_all):,}")
    print(f"  Range E: [{ps_all[:,6].min():.3f}, {ps_all[:,6].max():.3f}] MeV")
    print(f"  Range x: [{ps_all[:,0].min():.2f}, {ps_all[:,0].max():.2f}] cm")
    print(f"  Z const: {ps_all[:,2].mean():.2f} cm (std={ps_all[:,2].std():.4f})")

    return ps_all


def read_iaea_multi(
    phsp_paths: list,
    max_n_per_file: int = None,
    remove_511: bool = True,
    spatial_filter_cm: float = None,
) -> np.ndarray:
    """
    Legge e concatena più file .IAEAphsp.
    Ogni file viene parsato con il proprio header automaticamente.
    """
    all_ps = []
    for p in phsp_paths:
        print(f"\n{'─'*55}")
        ps = read_iaea_phsp(
            p,
            max_n=max_n_per_file,
            remove_511=remove_511,
            spatial_filter_cm=spatial_filter_cm,
        )
        all_ps.append(ps)

    combined = np.concatenate(all_ps, axis=0)
    print(f"\n{'='*55}")
    print(f"  Totale combinato: {len(combined):,} particelle da {len(phsp_paths)} file")
    return combined


def print_stats(ps: np.ndarray) -> None:
    cols = ["x [cm]", "y [cm]", "z [cm]", "dx", "dy", "dz", "E [MeV]"]
    print(f"\n  {'Canale':>10}  {'mu':>10}  {'σ':>10}  {'min':>10}  {'max':>10}")
    print(f"  {'-'*55}")
    for i, name in enumerate(cols):
        col = ps[:, i]
        print(f"  {name:>10}  {col.mean():>10.4f}  {col.std():>10.4f}  "
              f"{col.min():>10.4f}  {col.max():>10.4f}")

    print(f"\n  Correlazione E-dz: {np.corrcoef(ps[:,6], ps[:,5])[0,1]:.4f}  (paper: ~0.18)")
    print(f"  Correlazione x-dx: {np.corrcoef(ps[:,0], ps[:,3])[0,1]:.4f}  (paper: ~0.89)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Converte file IAEA phase space (.IAEAphsp) in HDF5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, nargs="+",
                   help="Uno o più file .IAEAphsp da leggere e concatenare")
    p.add_argument("--output", required=True,
                   help="Path del file HDF5 di output")
    p.add_argument("--max_n", type=int, default=None,
                   help="Numero massimo di record per file (None = tutto)")
    p.add_argument("--spatial_filter", type=float, default=None,
                   help="Filtra |x|,|y| < N cm (es. 7.5 per campo 15×15)")
    p.add_argument("--stats", action="store_true",
                   help="Stampa statistiche descrittive")
    p.add_argument("--remove_511", action="store_true", default=True,
                   help="Rimuovi picco annichilazione 511 keV")
    # Etichette per dataset condizionale
    p.add_argument("--E_nom", type=float, default=None,
                   help="Energia nominale del fascio [MeV] (per dataset condizionale)")
    p.add_argument("--jaw_x", type=float, default=None,
                   help="Semi-apertura jaw X [cm] (per dataset condizionale)")
    p.add_argument("--jaw_y", type=float, default=None,
                   help="Semi-apertura jaw Y [cm] (per dataset condizionale)")

    args = p.parse_args()

    from synthetic_linac import save_phase_space_hdf5

    # Lettura multi-file
    if len(args.input) > 1:
        ps = read_iaea_multi(
            args.input,
            max_n_per_file=args.max_n,
            remove_511=args.remove_511,
            spatial_filter_cm=args.spatial_filter,
        )
    else:
        ps = read_iaea_phsp(
            args.input[0],
            max_n=args.max_n,
            remove_511=args.remove_511,
            spatial_filter_cm=args.spatial_filter,
        )

    # Crea il vettore conditions se le etichette sono fornite
    conditions = None
    if args.E_nom is not None and args.jaw_x is not None and args.jaw_y is not None:
        conditions = np.tile(
            [args.E_nom, args.jaw_x, args.jaw_y],
            (len(ps), 1)
        ).astype(np.float32)
        print(f"\n  Conditions: E_nom={args.E_nom}, jaw_x={args.jaw_x}, jaw_y={args.jaw_y}")

    metadata = {
        "source": " + ".join(args.input),
        "n_particles": str(len(ps)),
    }
    if args.E_nom is not None:
        metadata["E_nom"] = str(args.E_nom)
    if args.jaw_x is not None:
        metadata["jaw_x"] = str(args.jaw_x)
    if args.jaw_y is not None:
        metadata["jaw_y"] = str(args.jaw_y)

    save_phase_space_hdf5(ps, conditions, args.output, metadata=metadata)
    print(f"\n  ✓ Salvato: {args.output}  ({len(ps):,} campioni)")

    if args.stats:
        print_stats(ps)