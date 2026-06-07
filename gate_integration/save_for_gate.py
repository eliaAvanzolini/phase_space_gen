"""
gate_integration/save_for_gate.py
==================================
Salva un modello addestrato (GAN, NSF o CFM) nel formato .pth compatibile
con GATE 10 GANSource.

Il formato .pth di gaga_phsp è un dizionario PyTorch che contiene:
    - "G_model_state":  state_dict del generatore
    - "G_model_str":    stringa JSON dell'architettura
    - "keys":           lista dei nomi delle colonne generate
    - "x_mean":         media delle colonne (per denormalizzazione)
    - "x_std":          deviazione standard delle colonne
    - "params":         dict con tutti i parametri del modello

Una volta salvato, il file viene usato in GATE 10 così:
    gsource = sim.add_source("GANSource", "linac")
    gsource.pth_filename = "linac_gan.pth"
    gsource.position_keys = ["X", "Y", "Z"]
    gsource.direction_keys = ["dX", "dY", "dZ"]
    gsource.energy_key = "Ekine"

Uso:
    # Salva GAN addestrato
    python gate_integration/save_for_gate.py \\
        --checkpoint outputs/gan_run/best_model.pt \\
        --model gan \\
        --stats_path outputs/gan_run/normalization_stats.json \\
        --out linac_6MV_gan.pth

    # Salva NSF (campionamento tramite wrapper GAN-compatibile)
    python gate_integration/save_for_gate.py \\
        --checkpoint outputs/nsf_run/best_model.pt \\
        --model nsf \\
        --stats_path outputs/nsf_run/normalization_stats.json \\
        --out linac_6MV_nsf.pth
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Colonne nel formato gaga_phsp (ordine GATE) ────────────────────────────────
# GATE si aspetta le chiavi nel .pth file per poterle associare a position/direction/energy
GATE_KEYS = ["Ekine", "X", "Y", "Z", "dX", "dY", "dZ"]
# Mapping dal nostro ordine interno (E, x, y, z, dx, dy, dz)
# al formato gaga (Ekine, X, Y, Z, dX, dY, dZ) — stessa struttura, nomi diversi

# Ordine delle colonne nel nostro normalizer 6D: (x, y, dx, dy, dz, E)
# Con z=0: (x, y, dx, dy, dz, E) → riordiniamo in (E, x, y, 0, dx, dy, dz)
OUR_TO_GATE = [5, 0, 1, None, 2, 3, 4]  # None = z=0 costante


def build_gate_compatible_generator(model, model_type, stats, device="cpu"):
    """
    Crea un wrapper che si comporta come il generatore di gaga_phsp.

    Il wrapper ha la stessa interfaccia del Generator di gaga_phsp:
        - forward(z) → tensor (B, 7) in spazio fisico (non normalizzato)
        - z_dim: dimensione latente

    Per NSF e CFM, wrappa il metodo sample() in un'interfaccia GAN-compatibile.
    """
    import torch
    import torch.nn as nn

    col_names = stats.get("col_names", ["x", "y", "dx", "dy", "dz", "E"])

    class GATECompatibleGenerator(nn.Module):
        """
        Wrapper che:
        1. Riceve z ~ N(0,I) come input
        2. Genera campioni nel spazio normalizzato
        3. Denormalizza → spazio fisico
        4. Riordina le colonne nel formato GATE: (E, x, y, z=0, dx, dy, dz)
        """
        def __init__(self, inner_model, model_type, stats_dict):
            super().__init__()
            self.inner = inner_model
            self.model_type = model_type
            self.stats = stats_dict
            self.z_dim = 6  # standard per tutti i modelli

        def forward(self, z):
            B = z.shape[0]
            device = z.device

            with torch.no_grad():
                if self.model_type == "gan":
                    # GAN: z → sample direttamente
                    s_norm = self.inner(z)
                elif self.model_type == "nsf":
                    # NSF: ignora z, campiona dalla distribuzione appresa
                    s_norm = self.inner.sample(B)
                elif self.model_type == "cfm":
                    # CFM: ignora z, usa Euler veloce
                    s_norm = self.inner.sample_fast(B, n_steps=10)
                else:
                    raise ValueError(f"Modello sconosciuto: {self.model_type}")

            # Denormalizza: spazio normalizzato → fisico
            s_phys = self._denormalize(s_norm.cpu().numpy())

            # Riordina in formato GATE: (E, x, y, z=0, dx, dy, dz)
            N = len(s_phys)
            gate_out = np.zeros((N, 7), dtype=np.float32)

            # Ordine interno: (x, y, dx, dy, dz, E) → indici 0,1,2,3,4,5
            gate_out[:, 0] = s_phys[:, 5]   # Ekine = E (col 5)
            gate_out[:, 1] = s_phys[:, 0]   # X = x (col 0)
            gate_out[:, 2] = s_phys[:, 1]   # Y = y (col 1)
            gate_out[:, 3] = 0.0            # Z = 0 (piano isocentrico)
            gate_out[:, 4] = s_phys[:, 2]   # dX = dx (col 2)
            gate_out[:, 5] = s_phys[:, 3]   # dY = dy (col 3)
            gate_out[:, 6] = s_phys[:, 4]   # dZ = dz (col 4)

            return torch.from_numpy(gate_out).to(device)

        def _denormalize(self, s_norm):
            """Inverte la normalizzazione."""
            col_names = self.stats.get("col_names", ["x", "y", "dx", "dy", "dz", "E"])
            s_phys = s_norm.copy()
            for i, col in enumerate(col_names):
                mu    = self.stats.get(f"{col}_mu", 0.0)
                sigma = self.stats.get(f"{col}_sigma", 1.0)
                s_phys[:, i] = s_norm[:, i] * sigma + mu
            return s_phys

    return GATECompatibleGenerator(model, model_type, stats)


def save_pth_for_gate(
    wrapper,
    stats: dict,
    out_path: str,
    params: dict = None,
):
    """
    Salva il modello nel formato .pth compatibile con gaga_phsp / GATE 10.

    Il formato è un dizionario con le chiavi richieste da gaga_phsp:
        G_model_state, G_model_str, keys, x_mean, x_std, params
    """
    import torch

    col_names_internal = stats.get("col_names", ["x", "y", "dx", "dy", "dz", "E"])

    # Media e std nel formato GATE (Ekine, X, Y, Z, dX, dY, dZ)
    key_map = {"E": 0, "x": 1, "y": 2, "z": 3, "dx": 4, "dy": 5, "dz": 6}
    x_mean = np.zeros(7, dtype=np.float32)
    x_std  = np.ones(7, dtype=np.float32)

    col_to_gate = {"E": 0, "x": 1, "y": 2, "dx": 4, "dy": 5, "dz": 6}
    for col in col_names_internal:
        if col in col_to_gate:
            idx = col_to_gate[col]
            x_mean[idx] = stats.get(f"{col}_mu",    0.0)
            x_std[idx]  = stats.get(f"{col}_sigma", 1.0)

    # Il generatore wrapper genera già output in spazio fisico
    # quindi mean=0 e std=1 per il file .pth (denorm già applicata internamente)
    x_mean_gate = np.zeros(7, dtype=np.float32)
    x_std_gate  = np.ones(7,  dtype=np.float32)

    if params is None:
        params = {"model_type": "phase_space_gen_wrapper"}

    # z_const_cm: piano sorgente (0=sintetico, 27.21=IAEA Elekta)
    params["z_const_cm"] = stats.get("z_const", 0.0)
    params["keys"] = GATE_KEYS
    params["gpu_mode"] = "auto"
    params["d_layers"] = 3
    params["g_layers"] = 3
    params["d_dim"] = 256
    params["g_dim"] = 256
    params["z_dim"] = 6
    params["d_learning_rate"] = 1e-5
    params["g_learning_rate"] = 1e-5
    params["batch_size"] = 16384
    params["epoch"] = 500
    params["optimiser"] = "adam"

    torch.save({
        "G_model_state": wrapper.state_dict(),
        "G_model_str":   json.dumps({"type": "wrapper", "keys": GATE_KEYS}),
        "keys":          GATE_KEYS,
        "x_mean":        x_mean_gate,
        "x_std":         x_std_gate,
        "params":        params,
        # Metadati aggiuntivi per nostra documentazione
        "_phase_space_gen": {
            "col_names":     col_names_internal,
            "original_mean": {c: float(stats.get(f"{c}_mu", 0)) for c in col_names_internal},
            "original_std":  {c: float(stats.get(f"{c}_sigma", 1)) for c in col_names_internal},
        }
    }, out_path)

    print(f"  Modello salvato per GATE 10: {out_path}")
    print(f"  Chiavi generate: {GATE_KEYS}")
    print(f"  Uso in GATE 10:")
    print(f"    gsource.pth_filename   = '{out_path}'")
    print(f"    gsource.energy_key     = 'Ekine'")
    print(f"    gsource.position_keys  = ['X', 'Y', 'Z']")
    print(f"    gsource.direction_keys = ['dX', 'dY', 'dZ']")


def load_model(checkpoint_path, model_type, device="cpu"):
    """Carica il modello dal checkpoint."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if model_type == "gan":
        from models.gan import PhaseSpaceGenerator
        sd = ckpt.get("G") or ckpt.get("generator") or ckpt
        cond_dim = 3 if any("cond" in k for k in sd.keys()) else 0
        model = PhaseSpaceGenerator(cond_dim=cond_dim)
        model.load_state_dict(sd)

    elif model_type == "nsf":
        from models.nsf import PhaseSpaceNSF
        sd = ckpt.get("model") or ckpt
        cond_dim = 3 if any("cond_encoder" in k for k in sd.keys()) else 0
        # Infer dim: spherical models have 5D, cartesian 6D
        # Check flow input size from first RandomPermutation
        dim = 6  # default
        for k, v in sd.items():
            if "permutation" in k and "perm" in k:
                dim = v.shape[0]; break
        # Fallback: check stats col_names length
        col_names = stats.get("col_names", [])
        if col_names: dim = len(col_names)
        model = PhaseSpaceNSF(dim=dim, cond_dim=cond_dim)
        model.load_state_dict(sd)

    elif model_type == "cfm":
        from models.cfm import PhaseSpaceCFM
        sd = ckpt.get("model") or ckpt
        cond_dim = 3 if any("cond_embed" in k for k in sd.keys()) else 0
        
        # ─── MODIFICA: Rilevamento dinamico di hidden_dim ────────────────────
        hidden_dim = 256  # Valore di default standard
        if "velocity_net.input_proj.weight" in sd:
            hidden_dim = sd["velocity_net.input_proj.weight"].shape[0]
            print(f"  [INFO] Rilevato hidden_dim dal checkpoint: {hidden_dim}")
        # ─────────────────────────────────────────────────────────────────────
        
        # Inseriamo hidden_dim nell'inizializzazione del modello
        model = PhaseSpaceCFM(dim=6, cond_dim=cond_dim, hidden_dim=hidden_dim)
        model.load_state_dict(sd)

    else:
        raise ValueError(f"Modello non riconosciuto: {model_type}")

    model.to(device).eval()
    return model


def main():
    p = argparse.ArgumentParser(
        description="Salva un modello nel formato .pth per GATE 10 GANSource",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",  required=True, help="Path al .pt del modello")
    p.add_argument("--model",       required=True, choices=["gan", "nsf", "cfm"])
    p.add_argument("--stats_path",  required=True, help="JSON delle statistiche di normalizzazione")
    p.add_argument("--out",         default="model_for_gate.pth", help="Output .pth")
    p.add_argument("--device",      default="cpu")
    args = p.parse_args()

    import torch

    with open(args.stats_path) as f:
        stats = json.load(f)

    print(f"\n  Caricamento {args.model.upper()} da: {args.checkpoint}")
    model = load_model(args.checkpoint, args.model, args.device)

    print("  Creazione wrapper GATE-compatibile...")
    wrapper = build_gate_compatible_generator(model, args.model, stats, args.device)

    # Test: genera un batch di prova
    z_test = torch.randn(100, 6)
    out_test = wrapper(z_test)
    print(f"  Test generazione: {out_test.shape}, range E=[{out_test[:,0].min():.3f}, {out_test[:,0].max():.3f}]MeV")

    save_pth_for_gate(
        wrapper, stats, args.out,
        params={"model_type": args.model, "checkpoint": args.checkpoint}
    )


if __name__ == "__main__":
    main()
