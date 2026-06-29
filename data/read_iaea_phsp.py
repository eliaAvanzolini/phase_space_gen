"""
data/read_iaea_phsp.py
=======================
Legge i file di phase space IAEA (.IAEAphsp) dell'Elekta 6MV
e li converte in HDF5. (Versione corretta per Record Length = 33 byte)
"""

import sys
import struct
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def read_iaea_phsp(
    phsp_path: str,
    max_n: int = None,
    remove_511: bool = True,
) -> np.ndarray:
    
    phsp_path = Path(phsp_path)
    record_size = 33  # Hardcoded dall'header per evitare errori
    
    file_size   = phsp_path.stat().st_size
    n_records   = file_size // record_size
    n_to_read   = min(n_records, max_n) if max_n else n_records

    print(f"  File: {phsp_path.name} ({file_size/1e9:.2f} GB)")
    print(f"  Records nel file (da 33 byte): {n_records:,}")
    print(f"  Da leggere:        {n_to_read:,}")

    chunk_size = 500_000
    results = []
    n_read   = 0
    n_kept   = 0

    with open(phsp_path, "rb") as f:
        while n_read < n_to_read:
            n_chunk = min(chunk_size, n_to_read - n_read)
            raw     = f.read(n_chunk * record_size)
            if not raw:
                break

            actual = len(raw) // record_size
            if actual == 0:
                break

            # Griglia di byte
            buf = np.frombuffer(raw[:actual * record_size], dtype=np.uint8)
            buf = buf.reshape(actual, record_size)

            # ESTRAZIONE CON OFFSET CORRETTI (Il segreto è qui!)
            # Byte 0: Tipo particella
            particle_type = np.frombuffer(buf[:, 0:1].tobytes(), dtype=np.int8)
            
            # Byte 1-5: Energia
            E_raw = np.frombuffer(buf[:, 1:5].tobytes(), dtype=np.float32)
            E_abs = np.abs(E_raw)
            
            # Byte 5-9: Posizione X
            X = np.frombuffer(buf[:, 5:9].tobytes(), dtype=np.float32)
            
            # Byte 9-13: Posizione Y
            Y = np.frombuffer(buf[:, 9:13].tobytes(), dtype=np.float32)
            
            # Byte 13-17: Direzione u (dx)
            u = np.frombuffer(buf[:, 13:17].tobytes(), dtype=np.float32)
            
            # Byte 17-21: Direzione v (dy)
            v = np.frombuffer(buf[:, 17:21].tobytes(), dtype=np.float32)
            
            # ATTENZIONE: In questo file da 33 byte, W non è salvato. 
            # I byte 21-25 contengono il "Weight", non W.
            # Dobbiamo calcolare W analiticamente: u^2 + v^2 + w^2 = 1
            w2 = 1.0 - (u**2 + v**2)
            w = np.sqrt(np.maximum(0.0, w2))

            # L'header dice che Z è costante a 27.21 cm
            Z = np.full(actual, 27.21, dtype=np.float32)

            # Il file è per il 99.5% fotoni (da header). Ignoriamo il type byte.
            valid  = (E_abs > 0.001) & (E_abs < 20.0)
            
            # 3. Rimuovi il picco di annichilazione (se richiesto)
            if remove_511:
                valid &= ~((E_abs > 0.505) & (E_abs < 0.520))
            valid &= (np.abs(X) <= 7.5) & (np.abs(Y) <= 7.5)
            ps = np.column_stack([
                X[valid], Y[valid], Z[valid],
                u[valid], v[valid], w[valid],
                E_abs[valid]
            ]).astype(np.float32)

            results.append(ps)
            n_read += actual
            n_kept += ps.shape[0]

            if n_read % 1_000_000 == 0 or n_read >= n_to_read:
                print(f"  Letti {n_read:>10,} / {n_to_read:,}  "
                      f"(kept {n_kept:,}, {100*n_kept/n_read:.1f}%)", end="\r")

    print()

    if not results:
        raise ValueError("Nessun dato valido letto dal file phase space.")

    ps_all = np.concatenate(results, axis=0)

    # Verifica e forza la normalizzazione del vettore direzione (||d||=1)
    d_norms = np.linalg.norm(ps_all[:, 3:6], axis=1)
    max_dev = np.abs(d_norms - 1.0).max()
    if max_dev > 0.01:
        norms = d_norms.reshape(-1, 1)
        ps_all[:, 3:6] /= np.where(norms > 0, norms, 1.0)

    print(f"\n  Totale particelle mantenute: {len(ps_all):,}")
    print(f"  Range E: [{ps_all[:,6].min():.3f}, {ps_all[:,6].max():.3f}] MeV")
    print(f"  Range x: [{ps_all[:,0].min():.2f}, {ps_all[:,0].max():.2f}] cm")
    
    return ps_all


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
    p = argparse.ArgumentParser()
    p.add_argument("--input",    required=True)
    p.add_argument("--output",   required=True)
    p.add_argument("--header",   default=None) # Ignorato, logica fissa
    p.add_argument("--max_n",    type=int, default=None)
    p.add_argument("--stats",    action="store_true")
    args = p.parse_args()

    from synthetic_linac import save_phase_space_hdf5

    ps = read_iaea_phsp(args.input, args.max_n)
    
    save_phase_space_hdf5(ps, None, args.output, metadata={"source": args.input})
    print(f"\n  ✓ Salvato: {args.output}  ({len(ps):,} campioni)")

    if args.stats:
        print_stats(ps)