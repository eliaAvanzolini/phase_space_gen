"""
data/dataset.py
===============
Dataset PyTorch per il phase space modeling.

Gestisce:
    - Caricamento da HDF5 (formato GATE-compatibile)
    - Normalizzazione / denormalizzazione
    - Split train / validation / test riproducibile
    - DataLoader con batching efficiente
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple, Dict

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch non disponibile. Solo numpy mode attivo.")

import h5py


class PhaseSpaceDataset:
    """
    Dataset per il phase space di sorgenti di fisica medica.

    Compatibile con PyTorch DataLoader quando torch è disponibile;
    altrimenti usabile come semplice array numpy.

    Parametri
    ---------
    hdf5_path    : path al file HDF5 (output di synthetic_linac.py o GATE)
    normalize    : se True, standardizza ogni canale a mu=0, sigma=1
    stats_path   : path JSON per salvare/caricare le statistiche di normalizzazione
    max_samples  : limite superiore sul numero di campioni caricati
    """

    CHANNEL_NAMES = ["x", "y", "z", "dx", "dy", "dz", "E"]
    DIM = 7

    def __init__(
        self,
        hdf5_path: str,
        normalize: bool = True,
        stats_path: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.hdf5_path = hdf5_path
        self.normalize = normalize

        # Caricamento dati
        with h5py.File(hdf5_path, "r") as f:
            ps = f["phase_space"][:]
            self.conditions = f["conditions"][:] if "conditions" in f else None

        if max_samples is not None:
            ps = ps[:max_samples]
            if self.conditions is not None:
                self.conditions = self.conditions[:max_samples]

        self.ps_raw = ps.astype(np.float32)

        # Normalizzazione
        if normalize:
            stats_loaded = False
            if stats_path and Path(stats_path).exists():
                with open(stats_path) as f:
                    self.stats = json.load(f)
                stats_loaded = True
            else:
                self.stats = self._compute_stats(self.ps_raw)
                if stats_path:
                    with open(stats_path, "w") as f:
                        json.dump(self.stats, f, indent=2)

            self.ps = self._normalize(self.ps_raw)
        else:
            self.ps = self.ps_raw
            self.stats = None

    def _compute_stats(self, ps: np.ndarray) -> Dict:
        stats = {}
        for i, col in enumerate(self.CHANNEL_NAMES):
            stats[f"{col}_mu"]    = float(ps[:, i].mean())
            stats[f"{col}_sigma"] = float(ps[:, i].std()) + 1e-8
        return stats

    def _normalize(self, ps: np.ndarray) -> np.ndarray:
        out = ps.copy()
        for i, col in enumerate(self.CHANNEL_NAMES):
            out[:, i] = (ps[:, i] - self.stats[f"{col}_mu"]) / self.stats[f"{col}_sigma"]
        return out

    def denormalize(self, ps_norm: np.ndarray) -> np.ndarray:
        """Inverte la normalizzazione su un array di campioni."""
        if self.stats is None:
            return ps_norm
        out = ps_norm.copy()
        for i, col in enumerate(self.CHANNEL_NAMES):
            out[:, i] = ps_norm[:, i] * self.stats[f"{col}_sigma"] + self.stats[f"{col}_mu"]
        return out

    def __len__(self):
        return len(self.ps)

    def __getitem__(self, idx):
        """
        Returns
        -------
        Se conditions disponibili: (ps_sample, condition)
        Altrimenti:                ps_sample
        """
        if TORCH_AVAILABLE:
            s = torch.from_numpy(self.ps[idx])
            if self.conditions is not None:
                c = torch.from_numpy(self.conditions[idx])
                return s, c
            return s
        else:
            if self.conditions is not None:
                return self.ps[idx], self.conditions[idx]
            return self.ps[idx]

    def get_splits(
        self,
        train_frac: float = 0.70,
        val_frac:   float = 0.15,
        seed: int = 42,
    ):
        """
        Restituisce indici per split train/val/test riproducibile.

        Returns
        -------
        (train_idx, val_idx, test_idx) come array numpy
        """
        n = len(self)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)

        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)

        train_idx = perm[:n_train]
        val_idx   = perm[n_train:n_train + n_val]
        test_idx  = perm[n_train + n_val:]

        return train_idx, val_idx, test_idx

    def get_arrays(
        self,
        split: Optional[str] = None,
        train_frac: float = 0.70,
        val_frac:   float = 0.15,
        seed: int = 42,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Restituisce i dati come array numpy per un determinato split.

        split : "train", "val", "test", o None (tutto il dataset)
        """
        if split is None:
            return self.ps, self.conditions

        tr_idx, val_idx, test_idx = self.get_splits(train_frac, val_frac, seed)
        idx_map = {"train": tr_idx, "val": val_idx, "test": test_idx}
        idx = idx_map[split]

        ps = self.ps[idx]
        c  = self.conditions[idx] if self.conditions is not None else None
        return ps, c

    def get_dataloader(
        self,
        split: str = "train",
        batch_size: int = 1024,
        shuffle: bool = True,
        num_workers: int = 4,
        seed: int = 42,
    ):
        """
        Crea un PyTorch DataLoader per un determinato split.
        Richiede PyTorch.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch non disponibile. Usa get_arrays() invece.")

        from torch.utils.data import TensorDataset

        ps, c = self.get_arrays(split, seed=seed)
        ps_t  = torch.from_numpy(ps)

        if c is not None:
            c_t      = torch.from_numpy(c)
            dataset  = TensorDataset(ps_t, c_t)
        else:
            dataset  = TensorDataset(ps_t)

        g = torch.Generator().manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            generator=g if shuffle else None,
        )


def normalize_conditions(
    conditions: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """
    Normalizza il vettore di condizione c = [E_nom, jaw_x, jaw_y].
    """
    stats = {
        "mu":    conditions.mean(axis=0).tolist(),
        "sigma": (conditions.std(axis=0) + 1e-8).tolist(),
    }
    c_norm = (conditions - np.array(stats["mu"])) / np.array(stats["sigma"])
    return c_norm.astype(np.float32), stats
