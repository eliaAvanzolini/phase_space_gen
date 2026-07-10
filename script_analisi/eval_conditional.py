"""
eval_conditional.py
====================
Valuta il CFM condizionato configurazione per configurazione
e stampa la tabella comparativa finale.

Uso:
    # Valuta CFM condizionato per ogni configurazione
    python eval_conditional.py \\
        --checkpoint outputs/cfm_cond_run/best_model.pt \\
        --stats      outputs/cfm_cond_run/normalization_stats.json \\
        --cond_stats outputs/cfm_cond_run/condition_stats.json \\
        --output_dir outputs/cfm_cond_run/eval_per_config

    # Tabella comparativa completa (tutti e 4 i modelli)
    python eval_conditional.py --comparison_table \\
        --gan_report  outputs/gan_run/eval/gan_report.json \\
        --nsf_report  outputs/nsf_run/eval/nsf_report.json \\
        --cfm_report  outputs/cfm_run/eval/cfm_report.json \\
        --cfm_cond_dir outputs/cfm_cond_run/eval_per_config
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def print_comparison_table(
    gan_path:      str = None,
    nsf_path:      str = None,
    cfm_path:      str = None,
    cfm_cond_path: str = None,
):
    """Stampa la tabella comparativa finale GAN / NSF / CFM / CFM-cond."""

    def load(path):
        if path and Path(path).exists():
            with open(path) as f:
                data = json.load(f)
            # Normalizza il formato gaga_baseline (keys diverse)
            if "mean_w1" in data:   # formato baseline_gaga.py
                cols = ["E","x","y","dx","dy","dz"]
                w1d  = {c: data.get(c, {}).get("w1", float("nan")) for c in cols}
                w1d["mean"] = data["mean_w1"]
                return {
                    "w1":           w1d,
                    "mmd":          float("nan"),
                    "separability": {"accuracy": data.get("separability",
                                    {}).get("accuracy", float("nan")),
                                     "std": data.get("separability",
                                    {}).get("std", float("nan"))},
                }
            return data
        return None

    models = {}
    if gan_path:  models["GAN (Sarrut 2019)"] = load(gan_path)
    if nsf_path:  models["NSF"]               = load(nsf_path)
    if cfm_path:  models["CFM"]               = load(cfm_path)

    # Per il CFM condizionato: media sui risultati per-config
    if cfm_cond_path and Path(cfm_cond_path).exists():
        with open(cfm_cond_path) as f:
            cond = json.load(f)
        w1_means = [v["w1"]["mean"] for v in cond.values()]
        seps     = [v["separability"]["accuracy"] for v in cond.values()]
        mmds     = [v["mmd"] for v in cond.values()]
        w1_E     = [v["w1"]["E"] for v in cond.values()]
        models["CFM-cond"] = {
            "w1":            {"mean": np.mean(w1_means), "E": np.mean(w1_E)},
            "mmd":           np.mean(mmds),
            "separability":  {"accuracy": np.mean(seps), "std": np.std(seps)},
        }

    if not models:
        print("[ERROR] Nessun report trovato.")
        return

    # Intestazione
    w = 14
    print(f"\n{'='*80}")
    print(f"  TABELLA COMPARATIVA FINALE — Phase Space Modeling")
    print(f"{'='*80}")
    print(f"  {'Modello':<12} {'W1 medio':>{w}} {'W1 (E)':>{w}} {'MMD²':>{w}} "
          f"{'Sep.':>{w}} {'Sep.std':>{w}}")
    print(f"  {'-'*72}")

    for name, r in models.items():
        if r is None:
            continue
        w1_mean = r["w1"]["mean"]
        w1_e    = r["w1"].get("E", float("nan"))
        mmd     = r.get("mmd", float("nan"))
        sep     = r["separability"]["accuracy"]
        sep_std = r["separability"]["std"]

        # Evidenzia il valore migliore (approssimazione)
        print(f"  {name:<12} {w1_mean:>{w}.6f} {w1_e:>{w}.6f} "
              f"{mmd:>{w}.6f} {sep:>{w}.4f} {sep_std:>{w}.4f}")

    print(f"{'='*80}")
    print(f"  (Sep=0.50 ottimo, Sep=1.00 fallimento; W1 e MMD² più bassi = migliore)")
    print()

    # Miglioramento relativo rispetto alla GAN
    if "GAN" in models and models["GAN"]:
        gan_w1 = models["GAN"]["w1"]["mean"]
        gan_sep = models["GAN"]["separability"]["accuracy"]
        print(f"  Miglioramento relativo vs GAN baseline:")
        for name, r in models.items():
            if name == "GAN" or r is None:
                continue
            w1_imp  = (gan_w1 - r["w1"]["mean"]) / gan_w1 * 100
            sep_imp = (gan_sep - r["separability"]["accuracy"]) / (gan_sep - 0.5) * 100
            print(f"    {name:<12}: W1 -{w1_imp:.1f}%  |  Sep -{sep_imp:.1f}% verso ottimo")

    print()


def run_eval(args):
    """Carica il modello e valuta per configurazione."""
    import torch
    from data.synthetic_linac import DEFAULT_CONFIGS
    from evaluate import evaluate_conditional

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Carica statistiche
    with open(args.stats) as f:
        stats = json.load(f)
    with open(args.cond_stats) as f:
        cond_stats = json.load(f)

    # Carica modello
    from models.cfm import PhaseSpaceCFM
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd   = ckpt.get("model") or ckpt
    cond_dim = 3 if any("cond_embed" in k for k in sd.keys()) else 0

    hidden_dim = 256
    n_layers   = 4
    for k, v in sd.items():
        if "input_proj.weight" in k:
            hidden_dim = v.shape[0]
        if "res_layers" in k:
            idx = int(k.split(".")[2])
            n_layers = max(n_layers, idx + 1)

    model = PhaseSpaceCFM(dim=6, cond_dim=cond_dim,
                          hidden_dim=hidden_dim, n_layers=n_layers)
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"  Modello: hidden_dim={hidden_dim}, n_layers={n_layers}, cond_dim={cond_dim}")

    results = evaluate_conditional(
        model=model,
        stats=stats,
        cond_stats=cond_stats,
        configs=args.configs or DEFAULT_CONFIGS,
        n_samples_per_config=args.n_samples,
        output_dir=args.output_dir,
        device=device,
    )

    # Salva JSON
    out_json = Path(args.output_dir) / "conditional_eval.json"
    print(f"\n  Risultati salvati: {out_json}")

    return results


def parse_args():
    p = argparse.ArgumentParser(
        description="Valutazione CFM condizionato + tabella comparativa",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # Valutazione per-config
    ev = sub.add_parser("eval", help="Valuta CFM condizionato per configurazione")
    ev.add_argument("--checkpoint",  required=True)
    ev.add_argument("--stats",       required=True, help="normalization_stats.json")
    ev.add_argument("--cond_stats",  required=True, help="condition_stats.json")
    ev.add_argument("--output_dir",  default="outputs/eval_conditional")
    ev.add_argument("--n_samples",   type=int, default=50_000)
    ev.add_argument("--configs",     default=None,
                    help="Opzionale: JSON {nome: {E_nom, jaw_x, jaw_y}}")

    # Tabella comparativa
    tb = sub.add_parser("table", help="Stampa tabella comparativa finale")
    tb.add_argument("--gan_report",   default=None, help="gan_report.json")
    tb.add_argument("--nsf_report",   default=None, help="nsf_report.json")
    tb.add_argument("--cfm_report",   default=None, help="cfm_report.json")
    tb.add_argument("--cfm_cond_dir", default=None,
                    help="Cartella con conditional_eval.json")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "eval":
        if args.configs:
            import json
            args.configs = json.load(open(args.configs))
        run_eval(args)

    elif args.command == "table":
        cfm_cond_json = None
        if args.cfm_cond_dir:
            cfm_cond_json = str(Path(args.cfm_cond_dir) / "conditional_eval.json")

        print_comparison_table(
            gan_path      = args.gan_report,
            nsf_path      = args.nsf_report,
            cfm_path      = args.cfm_report,
            cfm_cond_path = cfm_cond_json,
        )
