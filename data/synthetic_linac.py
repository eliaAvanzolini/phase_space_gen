"""
synthetic_linac.py
==================
Generatore sintetico di phase space per un acceleratore lineare medico (linac).

Modella la distribuzione p(s | c) dove:
    s = (x, y, z, dx, dy, dz, E)  -- vettore di stato della particella (7D)
    c = (E_nom, jaw_x, jaw_y)      -- parametri di configurazione del fascio

La distribuzione è costruita su basi fisiche:
    - Energia: spettro di bremsstrahlung approssimato come una mixture di Gaussiane
      troncate (picco di bassa energia + coda ad alta energia).
    - Posizione (x, y): distribuzione uniforme nel campo jaw con bordi smussati
      (convoluzione con Gaussiana, simula la penombra del collimatore).
    - Posizione z: fissa a 0 (piano isocentrico di riferimento).
    - Direzione (dx, dy, dz): piccola divergenza Gaussiana attorno all'asse del
      fascio (0, 0, 1), parametrizzata con angolo polare theta e azimutale phi.

Tutto in numpy puro: nessuna dipendenza da PyTorch, utilizzabile
anche senza GPU per debug e sviluppo locale.
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Tuple


# ─── Costanti fisiche ──────────────────────────────────────────────────────────
SPEED_OF_LIGHT = 299_792_458.0   # m/s
E_REST_ELECTRON = 0.511          # MeV, massa a riposo dell'elettrone
SSD = 100.0                      # cm, Source-to-Surface Distance standard


# ─── Configurazioni di default per un linac 6 MV FFF ──────────────────────────
DEFAULT_CONFIGS = {
    "6MV_5x5":   {"E_nom": 6.0, "jaw_x": 2.5, "jaw_y": 2.5},
    "6MV_10x10": {"E_nom": 6.0, "jaw_x": 5.0, "jaw_y": 5.0},
    "6MV_20x20": {"E_nom": 6.0, "jaw_x": 10.0, "jaw_y": 10.0},
    "4MV_10x10": {"E_nom": 4.0, "jaw_x": 5.0, "jaw_y": 5.0},
    "10MV_10x10":{"E_nom": 10.0, "jaw_x": 5.0, "jaw_y": 5.0},
}


def _bremsstrahlung_energy(
    n_samples: int,
    E_nom: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Campiona l'energia dei fotoni da uno spettro di bremsstrahlung sintetico.

    Lo spettro viene modellato come:
        p(E) ∝ w1 * N(mu1, sig1) + w2 * N(mu2, sig2)   (troncata a [E_min, E_nom])

    dove:
        - Il primo componente (bassa energia) rappresenta fotoni scatter e contaminazione
        - Il secondo (alta energia) rappresenta i fotoni primari del fascio
    """
    E_min = 0.01  # MeV, soglia minima di detection

    # Parametri dello spettro scalati con E_nom
    # (calibrati su spettri GATE pubblicati per linac Varian Clinac iX)
    mu1   = 0.05 * E_nom          # picco scatter (bassa energia)
    sig1  = 0.02 * E_nom
    mu2   = 0.35 * E_nom          # picco primari
    sig2  = 0.20 * E_nom
    w1, w2 = 0.25, 0.75           # pesi delle due componenti

    energies = np.empty(n_samples)
    n_filled = 0

    # Rejection sampling per rispettare i bounds fisici [E_min, E_nom]
    while n_filled < n_samples:
        n_needed = n_samples - n_filled
        batch    = int(n_needed * 1.5) + 100  # oversampling per efficienza

        # Selezione del componente per ogni campione
        component = rng.choice([0, 1], size=batch, p=[w1, w2])
        mus  = np.where(component == 0, mu1,  mu2)
        sigs = np.where(component == 0, sig1, sig2)

        e = rng.normal(mus, sigs)
        mask = (e >= E_min) & (e <= E_nom)

        n_accept = min(mask.sum(), n_needed)
        energies[n_filled:n_filled + n_accept] = e[mask][:n_accept]
        n_filled += n_accept

    return energies


def _beam_position(
    n_samples: int,
    jaw_x: float,
    jaw_y: float,
    rng: np.random.Generator,
    penumbra_sigma: float = 0.15,  # cm, sigma di penombra al jaw
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Campiona la posizione (x, y) dei fotoni al piano isocentrico.

    Distribuzione uniforme nel campo jaw (−jaw_x, jaw_x) × (−jaw_y, jaw_y)
    convolta con una Gaussiana per simulare la penombra del collimatore.
    Corrisponde a un sigmoide smussato ai bordi del campo.
    """
    # Uniforme nel campo
    x = rng.uniform(-jaw_x, jaw_x, n_samples)
    y = rng.uniform(-jaw_y, jaw_y, n_samples)

    # Aggiunta di piccola componente scatter (fotoni fuori campo, ~3% del totale)
    n_scatter = int(0.03 * n_samples)
    idx = rng.choice(n_samples, size=n_scatter, replace=False)
    x[idx] += rng.normal(0, jaw_x * 0.5, n_scatter)
    y[idx] += rng.normal(0, jaw_y * 0.5, n_scatter)

    # Penombra: piccolo jitter gaussiano ai bordi
    x += rng.normal(0, penumbra_sigma, n_samples)
    y += rng.normal(0, penumbra_sigma, n_samples)

    return x, y


def _beam_direction(
    n_samples: int,
    jaw_x: float,
    jaw_y: float,
    rng: np.random.Generator,
    source_distance: float = 100.0,  # cm, distanza sorgente-isocentro
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Campiona la direzione (dx, dy, dz) dei fotoni con vincolo ||d||=1.

    La divergenza angolare è proporzionale all'apertura del jaw e inversa
    alla distanza sorgente-isocentro (legge geometrica):
        sigma_theta ~ arctan(jaw / SSD)
    """
    # Divergenza angolare massima (semi-angolo al bordo del campo)
    sigma_x = np.arctan(jaw_x / source_distance) * 0.5
    sigma_y = np.arctan(jaw_y / source_distance) * 0.5

    # Angoli di deviazione dalla direzione principale (0, 0, 1)
    theta_x = rng.normal(0, sigma_x, n_samples)  # angolo in piano xz
    theta_y = rng.normal(0, sigma_y, n_samples)  # angolo in piano yz

    # Conversione in versore cartesiano (small angle: dz ≈ cos(theta))
    dx = np.sin(theta_x)
    dy = np.sin(theta_y)
    dz = np.sqrt(np.maximum(1.0 - dx**2 - dy**2, 1e-9))  # garantisce ||d||=1

    return dx, dy, dz


def generate_phase_space(
    n_samples: int,
    E_nom: float   = 6.0,
    jaw_x: float   = 5.0,
    jaw_y: float   = 5.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Genera un array di phase space di dimensione (n_samples, 7).

    Colonne: [x, y, z, dx, dy, dz, E]
    Unità:   [cm, cm, cm, adim, adim, adim, MeV]

    Parametri
    ---------
    n_samples : numero di particelle da generare
    E_nom     : energia nominale del fascio [MeV]
    jaw_x     : semi-apertura jaw in x [cm]
    jaw_y     : semi-apertura jaw in y [cm]
    seed      : seed per riproducibilità

    Returns
    -------
    ps : ndarray di shape (n_samples, 7)
    """
    rng = np.random.default_rng(seed)

    E        = _bremsstrahlung_energy(n_samples, E_nom, rng)
    x, y     = _beam_position(n_samples, jaw_x, jaw_y, rng)
    dx, dy, dz = _beam_direction(n_samples, jaw_x, jaw_y, rng)
    z        = np.zeros(n_samples)

    ps = np.column_stack([x, y, z, dx, dy, dz, E])
    return ps.astype(np.float32)


def generate_multi_condition_dataset(
    n_per_config: int,
    configs: Optional[dict] = None,
    seed: int = 42,
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Genera un dataset multi-condizione (Fase 3 della roadmap).

    Returns
    -------
    ps_all    : (N_total, 7) array di phase space
    c_all     : (N_total, 3) array di condizioni [E_nom, jaw_x, jaw_y]
    label_map : dizionario config_name -> indice numerico
    """
    if configs is None:
        configs = DEFAULT_CONFIGS

    all_ps, all_c = [], []
    label_map = {}

    rng_seed = seed
    for idx, (name, params) in enumerate(configs.items()):
        label_map[name] = idx
        ps = generate_phase_space(
            n_per_config,
            E_nom=params["E_nom"],
            jaw_x=params["jaw_x"],
            jaw_y=params["jaw_y"],
            seed=rng_seed,
        )
        c = np.tile(
            [params["E_nom"], params["jaw_x"], params["jaw_y"]],
            (n_per_config, 1)
        ).astype(np.float32)

        all_ps.append(ps)
        all_c.append(c)
        rng_seed += 1

    ps_all = np.concatenate(all_ps, axis=0)
    c_all  = np.concatenate(all_c,  axis=0)

    # Shuffle
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ps_all))
    ps_all = ps_all[perm]
    c_all  = c_all[perm]

    if save_path is not None:
        save_phase_space_hdf5(ps_all, c_all, save_path, label_map)
        print(f"Salvato dataset multi-condizione in {save_path}")

    return ps_all, c_all, label_map


def save_phase_space_hdf5(
    ps: np.ndarray,
    conditions: Optional[np.ndarray],
    path: str,
    metadata: Optional[dict] = None,
) -> None:
    """
    Salva il phase space in formato HDF5 (standard per simulazioni GATE).

    Struttura del file:
        /phase_space  (N, 7) float32  -- vettori di stato
        /conditions   (N, 3) float32  -- condizioni [E_nom, jaw_x, jaw_y]  (se presenti)
        /metadata     attrs           -- parametri di simulazione
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("phase_space", data=ps,
                              compression="gzip", compression_opts=4,
                              chunks=(min(10000, len(ps)), 7))
        ds.attrs["columns"] = ["x_cm", "y_cm", "z_cm", "dx", "dy", "dz", "E_MeV"]

        if conditions is not None:
            c_ds = f.create_dataset("conditions", data=conditions,
                                    compression="gzip", compression_opts=4)
            c_ds.attrs["columns"] = ["E_nom_MeV", "jaw_x_cm", "jaw_y_cm"]

        if metadata is not None:
            grp = f.create_group("metadata")
            for k, v in metadata.items():
                grp.attrs[k] = str(v)


def load_phase_space_hdf5(
    path: str,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Carica phase space da file HDF5.

    Returns (ps, conditions) dove conditions può essere None.
    """
    with h5py.File(path, "r") as f:
        ps = f["phase_space"][:]
        cond = f["conditions"][:] if "conditions" in f else None

    if max_samples is not None:
        ps   = ps[:max_samples]
        if cond is not None:
            cond = cond[:max_samples]

    return ps, cond


def _cartesian_to_spherical(dx, dy, dz):
    """
    Converte direzione cartesiana (dx,dy,dz) in coordinate sferiche (theta, phi).

    theta = arccos(dz)  ∈ [0, pi/2]  -- angolo polare dall'asse fascio
    phi   = atan2(dy,dx) ∈ [-pi, pi] -- angolo azimutale

    Vantaggi rispetto a (dx,dy,dz):
    - Nessun vincolo ||d||=1 da rispettare: è garantito per costruzione
    - Nessuna parete verticale a dz=1: theta=0 è un punto regolare
    - Distribuzione smooth senza discontinuità ai bordi fisici
    """
    dz_clipped = np.clip(dz, -1.0, 1.0)
    theta = np.arccos(dz_clipped).astype(np.float32)
    phi   = np.arctan2(dy, dx).astype(np.float32)
    return theta, phi


def _spherical_to_cartesian(theta, phi):
    """Inverte la trasformazione sferica → cartesiana."""
    sin_theta = np.sin(theta)
    dx = (sin_theta * np.cos(phi)).astype(np.float32)
    dy = (sin_theta * np.sin(phi)).astype(np.float32)
    dz = np.cos(theta).astype(np.float32)
    return dx, dy, dz
def _rank_transform(x: np.ndarray):
    """
    Applica la Probability Integral Transform a un array 1D.
    Restituisce (uniform_values, sorted_quantiles).
    Usa argsort invece di rankdata — molto più veloce su array grandi.
    """
    n = len(x)
    order = np.argsort(x)          # indici che ordinano x
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = (np.arange(n, dtype=np.float32) + 1.0) / (n + 1.0)
    sorted_q = x[order].astype(np.float32)  # quantili empirici ordinati
    return ranks, sorted_q


def _rank_inverse(u: np.ndarray, sorted_q: np.ndarray) -> np.ndarray:
    """
    Inverte la rank transform: mappa valori uniformi (0,1)
    ai valori originali tramite interpolazione sui quantili empirici.
    """
    n = len(sorted_q)
    # u è in (0,1) → indice continuo in [0, n-1]
    indices = np.clip(u * n, 0.0, n - 1.0)
    return np.interp(indices, np.arange(n), sorted_q).astype(np.float32)

def normalize_phase_space(
    ps: np.ndarray,
    stats: Optional[dict] = None,
    drop_z: bool = True,
    spherical: bool = True,
    rank_spatial: bool = False,
    stats_dir: Optional[str] = None,   # cartella dove salvare i .npy
) -> Tuple[np.ndarray, dict]:
    """
    Normalizza il phase space per il training.
    Se rank_spatial=True applica la Probability Integral Transform
    su x e y prima della standardizzazione gaussiana — risolve il
    problema dei bordi netti del jaw.
    I quantili (123M float) vengono salvati in stats_dir/*.npy,
    NON nel JSON.
    """
    from scipy.stats import norm as sp_norm

    if drop_z:
        x, y = ps[:, 0].copy(), ps[:, 1].copy()
        dx, dy, dz, E = ps[:, 3], ps[:, 4], ps[:, 5], ps[:, 6]
    else:
        x, y = ps[:, 0].copy(), ps[:, 1].copy()
        dx, dy, dz, E = ps[:, 3], ps[:, 4], ps[:, 5], ps[:, 6]

    # ── RANK TRANSFORM su x e y ───────────────────────────────────────
    if rank_spatial:
        if stats is None:
            # Fase training: calcola e salva i quantili
            x_uni, x_sorted = _rank_transform(x)
            y_uni, y_sorted = _rank_transform(y)
            # Mappa uniforme (0,1) → gaussiana N(0,1) con probit
            x = sp_norm.ppf(x_uni).astype(np.float32)
            y = sp_norm.ppf(y_uni).astype(np.float32)
            # Salva quantili come .npy (non nel JSON — troppo grandi)
            if stats_dir is not None:
                Path(stats_dir).mkdir(parents=True, exist_ok=True)
                np.save(str(Path(stats_dir) / "x_quantiles.npy"), x_sorted)
                np.save(str(Path(stats_dir) / "y_quantiles.npy"), y_sorted)
        else:
            # Fase inferenza/eval: carica quantili salvati
            q_dir = stats.get("quantiles_dir", stats_dir)
            x_sorted = np.load(str(Path(q_dir) / "x_quantiles.npy"))
            y_sorted = np.load(str(Path(q_dir) / "y_quantiles.npy"))
            # Mappa x,y sui quantili del training (searchsorted = O(N log N))
            x_uni = np.searchsorted(x_sorted, x).astype(np.float32) / len(x_sorted)
            y_uni = np.searchsorted(y_sorted, y).astype(np.float32) / len(y_sorted)
            x_uni = np.clip(x_uni, 1e-6, 1.0 - 1e-6)
            y_uni = np.clip(y_uni, 1e-6, 1.0 - 1e-6)
            x = sp_norm.ppf(x_uni).astype(np.float32)
            y = sp_norm.ppf(y_uni).astype(np.float32)

    # ── Il resto è IDENTICO alla versione originale ───────────────────
    if spherical:
        theta, phi = _cartesian_to_spherical(dx, dy, dz)
        ps_work   = np.column_stack([x, y, theta, phi, E]).astype(np.float32)
        col_names = ["x", "y", "theta", "phi", "E"]
    else:
        ps_work   = np.column_stack([x, y, dx, dy, dz, E]).astype(np.float32)
        col_names = ["x", "y", "dx", "dy", "dz", "E"]

    if stats is None:
        stats = {
            "drop_z": drop_z,
            "spherical": spherical,
            "col_names": col_names,
            "rank_spatial": rank_spatial,
        }
        if rank_spatial and stats_dir is not None:
            stats["quantiles_dir"] = str(stats_dir)
        if drop_z and ps.shape[1] == 7:
            z_col = ps[:, 2]
            if z_col.std() < 1e-3:
                stats["z_const"] = float(z_col.mean())
        for i, col in enumerate(col_names):
            stats[f"{col}_mu"]    = float(ps_work[:, i].mean())
            stats[f"{col}_sigma"] = float(ps_work[:, i].std()) + 1e-8

    ps_norm = np.empty_like(ps_work)
    for i, col in enumerate(col_names):
        ps_norm[:, i] = (ps_work[:, i] - stats[f"{col}_mu"]) / stats[f"{col}_sigma"]

    return ps_norm, stats

def denormalize_phase_space(ps_norm: np.ndarray, stats: dict) -> np.ndarray:
    """
    Inverte la normalizzazione. Se rank_spatial=True inverte anche
    la Probability Integral Transform su x e y.
    """
    from scipy.stats import norm as sp_norm

    col_names    = stats.get("col_names",    ["x", "y", "dx", "dy", "dz", "E"])
    drop_z       = stats.get("drop_z",       True)
    spherical    = stats.get("spherical",    False)
    z_const      = stats.get("z_const",      0.0)
    rank_spatial = stats.get("rank_spatial", False)

    # 1. Inversione standardizzazione gaussiana (identica a prima)
    ps_work = np.empty_like(ps_norm)
    for i, col in enumerate(col_names):
        ps_work[:, i] = ps_norm[:, i] * stats[f"{col}_sigma"] + stats[f"{col}_mu"]

    # 2. Inversione rank transform su x e y
    if rank_spatial:
        q_dir    = stats.get("quantiles_dir", ".")
        x_sorted = np.load(str(Path(q_dir) / "x_quantiles.npy"))
        y_sorted = np.load(str(Path(q_dir) / "y_quantiles.npy"))
        # Gaussiana N(0,1) → uniforme (0,1) → valori originali
        x_uni = sp_norm.cdf(ps_work[:, 0]).astype(np.float32)
        y_uni = sp_norm.cdf(ps_work[:, 1]).astype(np.float32)
        ps_work[:, 0] = _rank_inverse(x_uni, x_sorted)
        ps_work[:, 1] = _rank_inverse(y_uni, y_sorted)

    # 3. Ricostruzione vettore 7D (identica a prima)
    N = len(ps_work)
    ps_full = np.zeros((N, 7), dtype=np.float32)

    if spherical:
        x, y       = ps_work[:, 0], ps_work[:, 1]
        theta, phi = ps_work[:, 2], ps_work[:, 3]
        E          = ps_work[:, 4]
        dx, dy, dz = _spherical_to_cartesian(theta, phi)
        ps_full[:, 0] = x;  ps_full[:, 1] = y
        ps_full[:, 2] = z_const
        ps_full[:, 3] = dx; ps_full[:, 4] = dy; ps_full[:, 5] = dz
        ps_full[:, 6] = E
    else:
        ps_full[:, :2] = ps_work[:, :2]
        ps_full[:, 2]  = z_const
        ps_full[:, 3:] = ps_work[:, 2:]

    return ps_full
