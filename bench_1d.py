"""
bench_1d.py  -  Benchmark robusto del trasporto 1D dell'energia
===============================================================
Modelli a confronto:
  1. shift_k101       : Baseline fisica manuale (p=1.77, k=454.5)
  2. fit_pk           : Baseline fisica fittata con scipy
  3. resq (ResidualQ) : Fisica analitica + correzione residua MLP h(tau, d, Delta_d)
  4. ode (ODEv)       : Neural ODE du/dz = -Softplus(MLP(u, tau, z)) in spazio v=E^p
  5. deep_resq        : DeepRangeResQ 100% Neurale (LogRangeNet g_theta + StraggleNet h_phi)

Esperimenti:
  --exp pairs  : Mappa tutte le coppie sorgente -> bersaglio raggruppate per fascia di Delta_d.
  --exp energy : Test di generalita' in energia (training a bassa energia <= 40 MeV, test a 60 MeV).

Metriche:
  - W1/std          : Distanza di Wasserstein standardizzata.
  - tail_err        : Errore relativo medio della massa generata sotto i quantili 1, 2, 5, 10%.
  - dose_w1         : W1 pesato con il potere frenante approssimato S(E) = E^-0.77.
  - dose_mean_err   : Errore relativo sulla dose media S(E).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from full_ai_1d import TRAIN6, metrics, quantile_tables, train_pairs, save_rows
from robust_1d import ResQ, ODEv, fit_pk_window
from train_gen_energy_1d import load_proton_energies, make_quantile_grid, physics_transport
from deep_ai_1d import DeepRangeResQ, train_deep_resq

LEVELS = (0.01, 0.02, 0.05, 0.10)


def stopping_weight(E):
    return np.maximum(E, 0.3) ** (-0.77)


def extra_metrics(E_real, E_gen):
    m = metrics(E_real, E_gen)
    qs = np.quantile(E_real, LEVELS)
    rel = [(np.mean(E_gen <= q) - a) / a for q, a in zip(qs, LEVELS)]
    m["tail_err"] = float(np.mean(np.abs(rel)))
    wr, wg = stopping_weight(E_real), stopping_weight(E_gen)
    m["dose_mean_err"] = float((wg.mean() - wr.mean()) / wr.mean())
    m["dose_w1"] = float(wasserstein_distance(E_real, E_gen, wr, wg) / E_real.std())
    return m


def fit_model(kind, train_depths, E_by, tau, tau_np, dev, epochs, seed, no_ds=False, fit_max_depth=3.0):
    q_np, q_t = quantile_tables(E_by, train_depths, tau_np, dev)
    p, k = fit_pk_window(q_np, tau_np, None, max_depth=fit_max_depth)
    torch.manual_seed(seed)

    if kind == "resq":
        model = ResQ(p, k, 96, "no_dsrc" if no_ds else "full").to(dev)
        train_pairs(model, q_t, tau, epochs, lr=2e-3, wd=5e-3, max_dd=1.5, tag=kind, verbose=False)
    elif kind == "ode":
        model = ODEv(p, 96, use_z=not no_ds, use_tau=True).to(dev)
        train_pairs(model, q_t, tau, epochs, lr=3e-3, wd=1e-3, max_dd=1.5, tag=kind, verbose=False)
    elif kind == "deep_resq":
        model = DeepRangeResQ().to(dev)
        train_deep_resq(model, q_t, tau, warmup=max(150, epochs // 3), epochs=epochs, max_dd=1.5)
    else:
        raise ValueError(f"Modello sconosciuto: {kind}")

    return model, p, k


def line(name, m):
    return (f"  {name:14s} W1={m['w1']:.4f} tail_err={100 * m['tail_err']:5.1f}% KSt={m['ks']:.3f} "
            f"dose_w1={m['dose_w1']:.4f} dose_mean_err={100 * m['dose_mean_err']:+.1f}% dstd={m['dstd']:+.3f}")


def summarize(rows):
    bins = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 9.0)]
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    print("\n===== ERRORE PER FASCIA DI Delta d (media su tutte le coppie sorgente->bersaglio) =====")
    print(f"{'Delta d':>10s} {'metodo':14s} {'N':>3s} {'W1':>8s} {'tail_err%':>10s} {'dose_w1':>9s} {'dose_mean%':>11s}")
    for lo, hi in bins:
        for m in methods:
            sel = [r for r in rows if r["method"] == m and lo < r["delta"] <= hi + 1e-9]
            if not sel:
                continue
            print(f"{('%.1f-%.1f' % (lo, hi)):>10s} {m:14s} {len(sel):3d} "
                  f"{np.mean([r['w1'] for r in sel]):8.4f} {100 * np.mean([r['tail_err'] for r in sel]):10.1f} "
                  f"{np.mean([r['dose_w1'] for r in sel]):9.4f} {100 * np.mean([r['dose_mean_err'] for r in sel]):11.1f}")
        print()


def exp_pairs(a, E_by, tau, tau_np, dev):
    rows = []
    for t in [d for d in TRAIN6 if d > TRAIN6[0]]:
        train_d = [d for d in TRAIN6 if d != t]
        sources = [s for s in train_d if s < t]
        print(f"\n######## BERSAGLIO {t} cm, training su {train_d}, sorgenti {sources} ########")
        q_np, _ = quantile_tables(E_by, train_d, tau_np, dev)
        p, k = fit_pk_window(q_np, tau_np, None)

        # Addestramento di tutte e tre le reti senza la profondita' target t
        models = {
            kind: fit_model(kind, train_d, E_by, tau, tau_np, dev, a.epochs, 42)[0]
            for kind in ("resq", "ode", "deep_resq")
        }

        for s in sources:
            cases = {
                "shift_k101": physics_transport(E_by[s], s, t, 1.77, 450.0 * 1.01, 0.0),
                "fit_pk": physics_transport(E_by[s], s, t, p, k, 0.0),
            }
            for kind, m in models.items():
                cases[kind] = m.sample_from_source(E_by[s], s, t, tau)

            print(f" Sorgente {s} (Delta={t - s:.2f}):")
            for name, E in cases.items():
                mm = extra_metrics(E_by[t], E)
                rows.append(dict(target=t, source=s, delta=round(t - s, 2), method=name, **mm))
                print(line(name, mm))

    summarize(rows)
    save_rows(rows, f"{a.out}/pairs.csv")


def exp_energy(a, E_by, tau, tau_np, dev):
    train_d = [2.5, 3.0, 4.0]
    tests = [(0.5, 1.0), (1.0, 2.0), (0.5, 2.0)]
    q_np, _ = quantile_tables(E_by, train_d, tau_np, dev)
    p, k = fit_pk_window(q_np, tau_np, None, max_depth=4.0)
    print(f"[fit p,k su {train_d}] p={p:.3f} k={k:.1f}  (NB: la coppia con 4.0 contiene troncamento)")

    # Test di generalita': addestrate solo su energie basse <= 40 MeV
    models = {
        kind: fit_model(kind, train_d, E_by, tau, tau_np, dev, a.epochs, 42, no_ds=True, fit_max_depth=4.0)[0]
        for kind in ("resq", "ode", "deep_resq")
    }

    rows = []
    for s, t in tests:
        print(f"\n--- Trasporto {s} -> {t} cm (sorgente con E fino a ~{E_by[s].max():.0f} MeV; training max ~{max(E_by[d].max() for d in train_d):.0f} MeV) ---")
        cases = {
            "shift_k101": physics_transport(E_by[s], s, t, 1.77, 450.0 * 1.01, 0.0),
            "fit_pk(lowE)": physics_transport(E_by[s], s, t, p, k, 0.0),
        }
        for kind, m in models.items():
            cases[kind] = m.sample_from_source(E_by[s], s, t, tau)

        for name, E in cases.items():
            mm = extra_metrics(E_by[t], E)
            rows.append(dict(source=s, target=t, method=name, **mm))
            print(line(name, mm))

    save_rows(rows, f"{a.out}/energy.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/raw_phsp")
    ap.add_argument("--exp", choices=["pairs", "energy"], required=True)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--out", default="eval/bench_1d")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== DATI ===")
    E_by = load_proton_energies(a.data_dir, set(TRAIN6))
    tau = make_quantile_grid(2048, dev)
    tau_np = tau.cpu().numpy()
    Path(a.out).mkdir(parents=True, exist_ok=True)

    {"pairs": exp_pairs, "energy": exp_energy}[a.exp](a, E_by, tau, tau_np, dev)


if __name__ == "__main__":
    main()

