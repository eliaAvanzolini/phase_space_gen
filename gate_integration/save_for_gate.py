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
    Usa la denormalizzazione centralizzata del progetto per supportare nativamente
    sia le coordinate cartesiane 6D che quelle sferiche 5D.
    """
    import torch
    import torch.nn as nn
    from data.synthetic_linac import denormalize_phase_space

    class GATECompatibleGenerator(nn.Module):
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
                    s_norm = self.inner(z)
                elif self.model_type == "nsf":
                    s_norm = self.inner.sample(B)
                elif self.model_type == "cfm":
                    s_norm = self.inner.sample_fast(B, n_steps=10)
                else:
                    raise ValueError(f"Modello sconosciuto: {self.model_type}")

            # ── FIX: Usiamo la denormalizzazione ufficiale del progetto ──────
            # Restituisce SEMPRE una matrice 7D ordinata: [x, y, z, dx, dy, dz, E]
            s_phys = denormalize_phase_space(s_norm.cpu().numpy(), self.stats)

            N = len(s_phys)
            gate_out = np.zeros((N, 7), dtype=np.float32)

            # Riordina nel formato rigido richiesto da Geant4 / GATE 10
            gate_out[:, 0] = s_phys[:, 6]   # Ekine = E (colonna 6)
            gate_out[:, 1] = s_phys[:, 0]   # X = x (colonna 0)
            gate_out[:, 2] = s_phys[:, 1]   # Y = y (colonna 1)
            gate_out[:, 3] = s_phys[:, 2]   # Z = z_const (colonna 2)
            gate_out[:, 4] = s_phys[:, 3]   # dX = dx (colonna 3)
            gate_out[:, 5] = s_phys[:, 4]   # dY = dy (colonna 4)
            gate_out[:, 6] = s_phys[:, 5]   # dZ = dz (colonna 5)

            return torch.from_numpy(gate_out).to(device)

    return GATECompatibleGenerator(model, model_type, stats)


def load_model(checkpoint_path, model_type, device="cpu"):
    """Carica il modello dal checkpoint rilevando dinamicamente ogni iperparametro."""
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
        
        # 1. Rilevamento dinamico di dim (5D sferico o 6D cartesiano)
        dim = 6
        for k, v in sd.items():
            if "permutation" in k and "perm" in k:
                dim = v.shape[0]
                break
                
        # 2. Rilevamento dinamico di hidden_dim
        hidden_dim = 128
        for k, v in sd.items():
            if "transform_net.initial_layer.weight" in k:
                hidden_dim = v.shape[0]
                break
                
        # 3. Rilevamento dinamico di n_transforms
        max_idx = -1
        for k in sd.keys():
            if "_transforms." in k:
                idx_str = k.split("_transforms.")[1].split(".")[0]
                if idx_str.isdigit():
                    max_idx = max(max_idx, int(idx_str))
        n_transforms = (max_idx + 1) // 2 if max_idx >= 0 else 6
        
        # 4. Rilevamento dinamico di n_bins
        n_bins = 8
        for k, v in sd.items():
            if "transform_net.final_layer.bias" in k:
                out_feats = v.shape[0]
                for test_bins in [4, 8, 12, 16, 20]:
                    if out_feats % (3 * test_bins - 1) == 0:
                        n_bins = test_bins
                break

        model = PhaseSpaceNSF(dim=dim, cond_dim=cond_dim, n_transforms=n_transforms, hidden_dim=hidden_dim, n_bins=n_bins)
        model.load_state_dict(sd)

    elif model_type == "cfm":
        from models.cfm import PhaseSpaceCFM
        sd = ckpt.get("model") or ckpt
        cond_dim = 3 if any("cond_embed" in k for k in sd.keys()) else 0
        
        # 1. Rilevamento dinamico di dim
        dim = 6
        if "velocity_net.output_proj.weight" in sd:
            dim = sd["velocity_net.output_proj.weight"].shape[0]
            
        # 2. Rilevamento dinamico di hidden_dim
        hidden_dim = 256
        if "velocity_net.input_proj.weight" in sd:
            hidden_dim = sd["velocity_net.input_proj.weight"].shape[0]
            
        # 3. FIX: Rilevamento dinamico reale di n_layers (residual blocks)
        max_idx = -1
        for k in sd.keys():
            if "velocity_net.res_layers." in k:
                idx = int(k.split("res_layers.")[1].split(".")[0])
                max_idx = max(max_idx, idx)
        n_layers = (max_idx + 1) if max_idx >= 0 else 4

        model = PhaseSpaceCFM(dim=dim, cond_dim=cond_dim, n_layers=n_layers, hidden_dim=hidden_dim)
        model.load_state_dict(sd)

    else:
        raise ValueError(f"Modello non riconosciuto: {model_type}")

    model.to(device).eval()
    return model


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
