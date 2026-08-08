import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

MODELS = {
    "cfm": {
        "checkpoint": "outputs/cfm_conditional_6mv_10mv/best_model.pt",
        "stats_json": "outputs/cfm_conditional_6mv_10mv/normalization_stats.json",
        "model_type": "cfm",
    },
    "nsf": {
        "checkpoint": "outputs/nsf_conditional_6mv_10mv/best_model.pt",
        "stats_json": "outputs/nsf_conditional_6mv_10mv/normalization_stats.json",
        "model_type": "nsf",
    },
}

TRAIN_H5_PATH = "data/conditional_jaws_dataset.h5"  # dataset di training bilanciato (phase_space + conditions)


def _load_cfm(checkpoint_path, dim, device):
    from models.cfm import CFMTrainer, PhaseSpaceCFM
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model") or ckpt
    hidden_dim = 256
    if "velocity_net.input_proj.weight" in sd:
        hidden_dim = sd["velocity_net.input_proj.weight"].shape[0]
    n_layers = 4
    max_idx = -1
    for k in sd:
        if "velocity_net.res_layers." in k:
            max_idx = max(max_idx, int(k.split("res_layers.")[1].split(".")[0]))
    if max_idx >= 0:
        n_layers = max_idx + 1
    model = PhaseSpaceCFM(dim=dim, cond_dim=3, hidden_dim=hidden_dim, n_layers=n_layers)
    trainer = CFMTrainer(model, device=device, lr=1e-4)
    trainer.load(checkpoint_path)
    return model.to(device).eval()


def _load_nsf(checkpoint_path, dim, device):
    from models.nsf import NSFTrainer, PhaseSpaceNSF
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model") or ckpt
    hidden_dim = 256
    for k, v in sd.items():
        if "transform_net.initial_layer.weight" in k:
            hidden_dim = v.shape[0]
            break
    n_transforms = 6
    max_idx = -1
    for k in sd:
        if "_transforms." in k:
            s = k.split("_transforms.")[1].split(".")[0]
            if s.isdigit():
                max_idx = max(max_idx, int(s))
    if max_idx >= 0:
        n_transforms = (max_idx + 1) // 2
    model = PhaseSpaceNSF(dim=dim, cond_dim=3, n_transforms=n_transforms, hidden_dim=hidden_dim, n_bins=16, tail_bound=7.0)
    trainer = NSFTrainer(model, device=device, lr=1e-4)
    trainer.load(checkpoint_path)
    return model.to(device).eval()


def generate_at_condition(model_name, cond_vector, n_samples, device):
    """Genera phase space per una condizione arbitraria (anche mai vista in training)."""
    from data.synthetic_linac import denormalize_phase_space

    cfg = MODELS[model_name]
    with open(cfg["stats_json"]) as f:
        stats = json.load(f)
    dim = len(stats.get("col_names", ["x", "y", "theta", "phi", "E"]))

    cond_stats_path = Path(cfg["stats_json"]).parent / "condition_stats.json"
    with open(cond_stats_path) as f:
        cond_stats = json.load(f)
    mu_c = np.array(cond_stats["mu"], dtype=np.float32)
    sig_c = np.array(cond_stats["sigma"], dtype=np.float32)
    cond_norm = ((np.array(cond_vector, dtype=np.float32) - mu_c) / sig_c).tolist()

    is_cfm = cfg["model_type"] == "cfm"
    model = _load_cfm(cfg["checkpoint"], dim, device) if is_cfm else _load_nsf(cfg["checkpoint"], dim, device)

    chunk_size = 200_000
    chunks = []
    n_done = 0
    with torch.no_grad():
        while n_done < n_samples:
            n_chunk = min(chunk_size, n_samples - n_done)
            cond_tensor = torch.tensor(cond_norm, dtype=torch.float32, device=device).unsqueeze(0).repeat(n_chunk, 1)
            if is_cfm:
                chunk_t = model.sample(n_chunk, cond_tensor, method="dopri5", atol=1e-4, rtol=1e-4)
            else:
                chunk_t = model.sample(n_chunk, cond_tensor)
            chunks.append(chunk_t.cpu().numpy())
            n_done += n_chunk
            del cond_tensor, chunk_t
            if device == "cuda":
                torch.cuda.empty_cache()

    ps_norm = np.concatenate(chunks, axis=0)
    ps_out = denormalize_phase_space(ps_norm, stats).astype(np.float32)
    print(f"  [{model_name}] shape output denormalizzato: {ps_out.shape} "
          f"(atteso: (N, 7) se Z viene reinserita; z_const nello stats = {stats.get('z_const', 'ASSENTE')})")
    return ps_out


def load_real_condition(train_h5_path, cond_vector, atol=0.1):
    """Carica dal dataset di training i vettori reali corrispondenti a una data condizione."""
    with h5py.File(train_h5_path, "r") as f:
        ps_all = f["phase_space"][:]
        cond_all = f["conditions"][:]
    mask = np.all(np.abs(cond_all - np.array(cond_vector)) < atol, axis=1)
    return ps_all[mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", type=float, default=8.0, help="Energia target (MeV), anche mai vista in training")
    ap.add_argument("--jaw_x", type=float, default=5.0)
    ap.add_argument("--jaw_y", type=float, default=5.0)
    ap.add_argument("--n_samples", type=int, default=200_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_dir", default="outputs/energy_interpolation")
    args = ap.parse_args()

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cond_vector = [args.energy, args.jaw_x, args.jaw_y]
    print(f"Condizione target (interpolata): E={args.energy} MeV, jaw=({args.jaw_x}x{args.jaw_y}) cm")

    # ── Generazione dai modelli alla condizione mai vista ──────────────────
    generated = {}
    for model_name in MODELS:
        print(f"\nGenerazione con {model_name.upper()}...")
        ps = generate_at_condition(model_name, cond_vector, args.n_samples, device)
        generated[model_name] = ps
        np.save(out / f"generated_{model_name}_E{args.energy}.npy", ps)
        print(f"  {len(ps):,} vettori generati e salvati.")

    # ── Riferimenti reali alle energie di training (stesso jaw) ────────────
    print("\nCaricamento riferimenti reali dal dataset di training (6MV e 10MV, stesso jaw)...")
    real_6mv = load_real_condition(TRAIN_H5_PATH, [6.0, args.jaw_x, args.jaw_y])
    real_10mv = load_real_condition(TRAIN_H5_PATH, [10.0, args.jaw_x, args.jaw_y])
    print(f"  6MV reale: {len(real_6mv):,} vettori | 10MV reale: {len(real_10mv):,} vettori")

    if len(real_6mv) == 0 or len(real_10mv) == 0:
        print("⚠️ ATTENZIONE: uno dei due riferimenti reali e' vuoto. Controlla jaw_x/jaw_y "
              "e la tolleranza di match nel dataset di training (potrebbe non esserci "
              "esattamente questa combinazione bilanciata).")

    # Indice colonna energia: assumendo lo schema [X,Y,Z,dX,dY,dZ,E] usato nel resto della pipeline
    E_COL = 6

    # ── Statistiche riassuntive e check di monotonia ────────────────────────
    print(f"\n{'='*70}\nSTATISTICHE SPETTRO ENERGETICO (colonna E, indice {E_COL})\n{'='*70}")
    stats_summary = {}
    if len(real_6mv) > 0:
        e6 = real_6mv[:, E_COL]
        stats_summary["real_6mv"] = {"mean": float(e6.mean()), "median": float(np.median(e6)), "std": float(e6.std())}
        print(f"  REAL 6MV : mean={e6.mean():.4f}  median={np.median(e6):.4f}  std={e6.std():.4f}")
    if len(real_10mv) > 0:
        e10 = real_10mv[:, E_COL]
        stats_summary["real_10mv"] = {"mean": float(e10.mean()), "median": float(np.median(e10)), "std": float(e10.std())}
        print(f"  REAL 10MV: mean={e10.mean():.4f}  median={np.median(e10):.4f}  std={e10.std():.4f}")
    for model_name, ps in generated.items():
        eg = ps[:, E_COL]
        stats_summary[f"gen_{model_name}_{args.energy}MV"] = {
            "mean": float(eg.mean()), "median": float(np.median(eg)), "std": float(eg.std())
        }
        print(f"  GEN {model_name.upper()} @{args.energy}MV: mean={eg.mean():.4f}  median={np.median(eg):.4f}  std={eg.std():.4f}")

    if len(real_6mv) > 0 and len(real_10mv) > 0:
        print(f"\n  Check di monotonia (atteso: mean(6MV) < mean(gen) < mean(10MV)):")
        for model_name, ps in generated.items():
            eg_mean = ps[:, E_COL].mean()
            ok = e6.mean() < eg_mean < e10.mean()
            print(f"    {model_name.upper()}: {e6.mean():.4f} < {eg_mean:.4f} < {e10.mean():.4f}  "
                  f"-> {'OK monotono' if ok else '⚠️ NON monotono'}")

    # ── Check sull'endpoint bremsstrahlung (vincolo fisico piu' specifico) ──
    # Il taglio ad alta energia dello spettro fotonico e' vincolato dall'energia
    # nominale dell'elettrone incidente sul target - un check molto piu'
    # specifico della semplice media, perche' non dipende dalla forma
    # complessiva ma da un singolo punto fisico ben definito.
    print(f"\n{'='*70}\nCHECK ENDPOINT BREMSSTRAHLUNG (percentile 99.5 come proxy robusto del taglio)\n{'='*70}")
    if len(real_6mv) > 0:
        p995_6 = np.percentile(real_6mv[:, E_COL], 99.5)
        print(f"  REAL 6MV  P99.5 = {p995_6:.3f} MeV")
    if len(real_10mv) > 0:
        p995_10 = np.percentile(real_10mv[:, E_COL], 99.5)
        print(f"  REAL 10MV P99.5 = {p995_10:.3f} MeV")
    for model_name, ps in generated.items():
        p995_gen = np.percentile(ps[:, E_COL], 99.5)
        pos = "tra 6 e 10 (atteso)" if (len(real_6mv) and len(real_10mv) and p995_6 < p995_gen < p995_10) else "FUORI dall'intervallo atteso"
        print(f"  GEN {model_name.upper()} P99.5 = {p995_gen:.3f} MeV  -> {pos}")

    with open(out / "spectrum_stats_summary.json", "w") as f:
        json.dump(stats_summary, f, indent=2)

    # ── Plot: istogrammi energia sovrapposti ────────────────────────────────
    plt.figure(figsize=(9, 6))
    bins = np.linspace(0, max(
        real_6mv[:, E_COL].max() if len(real_6mv) else 1,
        real_10mv[:, E_COL].max() if len(real_10mv) else 1,
        *(ps[:, E_COL].max() for ps in generated.values()),
    ), 150)
    if len(real_6mv) > 0:
        plt.hist(real_6mv[:, E_COL], bins=bins, histtype="step", density=True, label="REAL 6MV", color="black", linestyle="--", linewidth=1.5)
    if len(real_10mv) > 0:
        plt.hist(real_10mv[:, E_COL], bins=bins, histtype="step", density=True, label="REAL 10MV", color="gray", linestyle="--", linewidth=1.5)
    colors = {"cfm": "tab:blue", "nsf": "tab:orange"}
    for model_name, ps in generated.items():
        plt.hist(ps[:, E_COL], bins=bins, histtype="step", density=True,
                  label=f"GEN {model_name.upper()} @{args.energy}MV", color=colors.get(model_name), linewidth=2)
    plt.xlabel("Energia (MeV)")
    plt.ylabel("Densita' (normalizzata)")
    plt.title(f"Spettro energetico: interpolazione a {args.energy}MV vs riferimenti reali\njaw=({args.jaw_x}x{args.jaw_y})cm")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / f"spectrum_comparison_E{args.energy}.png", dpi=130)
    print(f"\nPlot salvato in: {out / f'spectrum_comparison_E{args.energy}.png'}")
    print(f"Riepilogo statistiche in: {out / 'spectrum_stats_summary.json'}")


if __name__ == "__main__":
    main()
