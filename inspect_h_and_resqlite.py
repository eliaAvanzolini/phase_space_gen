"""
inspect_h_and_resqlite.py
=========================
1. Genera il plot di h(tau) per ResidualQ sui tre fold LOO:
   verifica se la correzione e' solo un offset k(d) o una forma di straggling fine.
2. Implementa e valuta 'ResQ-lite':
   correzione analitica a soli 2 parametri k_eff = k0 * (1 + eps * d).
   Confronta i fold LOO con ResidualQ e shift_k101.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
import torch

from full_ai_1d import TRAIN6, FOLDS, metrics, fmt, quantile_tables
from robust_1d import ResQ, fit_pk_window, train_pairs
from train_gen_energy_1d import load_proton_energies, make_quantile_grid, physics_transport


def fit_resq_lite(E_by, train_depths, p_fit, k_fit):
    """Fitta un singolo parametro eps per k_eff(d) = k0 * (1 + eps * d) su tutte le coppie note."""
    depths = sorted(train_depths)
    pairs = [(d1, d2) for d1 in depths for d2 in depths if 0 < d2 - d1 <= 1.5]

    def loss_fn(eps):
        err = 0.0
        for d1, d2 in pairs:
            # k medio attraversato tra d1 e d2
            d_mid = 0.5 * (d1 + d2)
            k_eff = k_fit * (1.0 + eps * (d_mid / 4.0))
            pred = physics_transport(E_by[d1], d1, d2, p_fit, k_eff, 0.0)
            qs = np.linspace(0.01, 0.99, 200)
            q_pred = np.quantile(pred, qs)
            q_real = np.quantile(E_by[d2], qs)
            err += np.mean(np.abs(q_pred - q_real))
        return err / len(pairs)

    res = minimize_scalar(loss_fn, bounds=(-0.15, +0.15), method="bounded")
    return float(res.x)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== 1. ISPEZIONE DI h(tau) E BENCHMARK RESQ-LITE ===")
    E_by = load_proton_energies("data/raw_phsp", set(TRAIN6))
    tau = make_quantile_grid(2048, dev)
    tau_np = tau.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    tau_eval = torch.linspace(1e-3, 1.0 - 1e-3, 500, device=dev)
    tau_eval_np = tau_eval.cpu().numpy()

    print("\n--- RISULTATI CONFRONTO SUI FOLD LEAVE-ONE-OUT ---")
    results = []

    for hold, src in FOLDS:
        train_d = [d for d in TRAIN6 if d != hold]
        q_np, q_t = quantile_tables(E_by, train_d, tau_np, dev)
        p, k = fit_pk_window(q_np, tau_np, None)

        # 1. Shift Fisico (baseline manuale)
        m_sh = metrics(E_by[hold], physics_transport(E_by[src], src, hold, 1.77, 450.0 * 1.01, 0.0))

        # 2. ResQ-lite (2 parametri: k0 ed eps di profondita')
        eps_opt = fit_resq_lite(E_by, train_d, p, k)
        k_eff = k * (1.0 + eps_opt * (0.5 * (src + hold) / 4.0))
        m_lite = metrics(E_by[hold], physics_transport(E_by[src], src, hold, p, k_eff, 0.0))

        # 3. ResidualQ (Rete Neurale completa h(tau, d, Delta_d))
        torch.manual_seed(42)
        rq = ResQ(p, k, hidden=96, feat="full").to(dev)
        train_pairs(rq, q_t, tau, epochs=600, lr=2e-3, wd=5e-3, max_dd=1.5, tag="resq", verbose=False)
        m_rq = metrics(E_by[hold], rq.sample_from_source(E_by[src], src, hold, tau))

        results.append(dict(fold=hold, shift=m_sh["w1"], lite=m_lite["w1"], resq=m_rq["w1"],
                            ks_sh=m_sh["ks"], ks_lite=m_lite["ks"], ks_rq=m_rq["ks"]))

        print(f"\nFold Hold-out {hold} cm (da {src} cm):")
        print(f"  shift_k101 : {fmt(m_sh)}")
        print(f"  ResQ-lite  : {fmt(m_lite)}  (eps={eps_opt:+.4f}, k_eff={k_eff:.1f})")
        print(f"  ResidualQ  : {fmt(m_rq)}  (Rete neurale MLP)")

        # Estrazione della curva h(tau)
        with torch.no_grad():
            dd = float(hold - src)
            inp = torch.stack([tau_eval,
                               torch.full_like(tau_eval, src / 4.0),
                               torch.full_like(tau_eval, dd / 4.0)], dim=-1)
            # h(tau) espressa in MeV equivalenti
            h_vals = rq.net(inp).squeeze(-1).cpu().numpy() * 500.0 * dd
            ax.plot(tau_eval_np, h_vals, label=f"Hold {hold} cm (da {src} cm, Delta={dd:.1f})", linewidth=1.8)

    ax.axhline(0, color="black", linestyle=":", alpha=0.6)
    ax.set_title("Cosa impara la rete neurale h(tau) in ResidualQ?", fontsize=12)
    ax.set_xlabel("Percentile energetico tau (0 = protoni lenti, 1 = protoni veloci)", fontsize=11)
    ax.set_ylabel("Correzione delta_v applicata dalla rete [MeV^p]", fontsize=11)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    Path("eval").mkdir(exist_ok=True)
    out_png = "eval/ispezione_h_tau.png"
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"\nGrafico salvato in: {out_png}")

    print("\n===== TABELLA RIASSUNTIVA MEDIE LEAVE-ONE-OUT =====")
    print(f"{'Metodo':18s} | {'W1/sd medio':>12s} | {'KS coda medio':>14s}")
    w_sh = np.mean([r["shift"] for r in results])
    w_lt = np.mean([r["lite"] for r in results])
    w_rq = np.mean([r["resq"] for r in results])
    k_sh = np.mean([r["ks_sh"] for r in results])
    k_lt = np.mean([r["ks_lite"] for r in results])
    k_rq = np.mean([r["ks_rq"] for r in results])
    print(f"{'shift_k101':18s} | {w_sh:12.4f} | {k_sh:14.4f}")
    print(f"{'ResQ-lite (2 param)':18s} | {w_lt:12.4f} | {k_lt:14.4f}")
    print(f"{'ResidualQ (Rete)':18s} | {w_rq:12.4f} | {k_rq:14.4f}")


if __name__ == "__main__":
    main()
