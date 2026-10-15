"""
bench_sde_1d.py
===============
Punto 3: Trasporto 1D dell'energia stocastico continuo (SDE di stato, senza tau).
  du = -a_phi(u) * dz + sigma_psi(u) * dW_z
Nessun rango tau, nessuna costante fisica cablata.
Esegue:
  --exp pairs  (mappa delle distanze Delta_d)
  --exp energy (estrapolazione a 60 MeV dopo training a <= 40 MeV)
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

from full_ai_1d import TRAIN6, metrics, quantile_tables, save_rows
from train_gen_energy_1d import load_proton_energies, make_quantile_grid, physics_transport
from deep_ai_1d import LogRangeNet


class NeuralStateSDE1D(nn.Module):
    """SDE differenziabile nello spazio latente u = g(E): du = -a(u)dz + sigma(u) dW."""
    def __init__(self, hidden=64):
        super().__init__()
        self.range_net = LogRangeNet(hidden=48)
        self.drift_net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )
        self.diff_net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )
        # Deriva a(u) ~ 1.0 (traslazione media unitaria per cm)
        nn.init.zeros_(self.drift_net[-1].weight)
        nn.init.constant_(self.drift_net[-1].bias, 0.5413)  # softplus(0.5413) = 1.0
        # Diffusione iniziale modesta
        nn.init.zeros_(self.diff_net[-1].weight)
        nn.init.constant_(self.diff_net[-1].bias, -2.0)     # softplus(-2) ~ 0.12

    def drift(self, u):
        return F.softplus(self.drift_net(u[:, None] / 4.0).squeeze(-1))

    def diffusion(self, u):
        return F.softplus(self.diff_net(u[:, None] / 4.0).squeeze(-1))

    def forward_sde(self, u0, dz, n_steps=12):
        dt = float(dz) / n_steps
        sqrt_dt = np.sqrt(dt)
        u = u0
        for _ in range(n_steps):
            u_clamped = u.clamp(min=0.0)
            a = self.drift(u_clamped)
            sig = self.diffusion(u_clamped)
            dw = torch.randn_like(u) * sqrt_dt
            u = u - a * dt + sig * dw
        return u

    def transport_samples(self, E_src_t, dz, n_steps=12):
        u_tab, _ = self.range_net.g_table()
        u0 = self.range_net.forward_E_to_u(E_src_t, u_tab)
        u_tgt = self.forward_sde(u0, dz, n_steps=n_steps)
        surv = u_tgt > 0.0
        E_tgt = self.range_net.inverse_u_to_E(u_tgt[surv], u_tab)
        return E_tgt


def train_sde(model, q_dict, tau, epochs=400, max_dd=1.5, device="cuda"):
    depths = sorted(q_dict.keys())
    pairs = [(a, b) for a in depths for b in depths if 0 < b - a <= max_dd + 1e-9]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    for ep in range(1, epochs + 1):
        opt.zero_grad()
        tot = 0.0
        for d1, d2 in pairs:
            # Campionamento di quantili ordinati per simulazione differenziabile
            E_src_q = q_dict[d1]
            E_pred = model.transport_samples(E_src_q, d2 - d1, n_steps=10)
            if len(E_pred) < 50:
                continue
            # Loss sui quantili interpolati
            q_pred = torch.quantile(E_pred, tau)
            loss_pair = torch.mean(torch.abs(q_pred - q_dict[d2]))
            tot = tot + loss_pair

        loss = tot / len(pairs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
    return model


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/raw_phsp")
    ap.add_argument("--exp", choices=["pairs", "energy"], required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--out", default="eval/sde_1d")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    E_by = load_proton_energies(a.data_dir, set(TRAIN6))
    tau = make_quantile_grid(2048, dev)
    tau_np = tau.cpu().numpy()
    Path(a.out).mkdir(parents=True, exist_ok=True)

    if a.exp == "pairs":
        print("=== BENCHMARK SDE 1D DI STATO: ESPERIMENTO PAIRS ===")
        rows = []
        for t in [d for d in TRAIN6 if d > TRAIN6[0]]:
            train_d = [d for d in TRAIN6 if d != t]
            sources = [s for s in train_d if s < t]
            print(f"\n######## BERSAGLIO {t} cm, training su {train_d} ########")
            q_np, q_t = quantile_tables(E_by, train_d, tau_np, dev)
            model = NeuralStateSDE1D().to(dev)
            train_sde(model, q_t, tau, epochs=a.epochs, max_dd=1.5, device=dev)
            model.eval()

            for s in sources:
                with torch.no_grad():
                    e_src_t = torch.from_numpy(E_by[s]).float().to(dev)
                    # Campiona fino a 300k protoni per la metrica
                    if len(e_src_t) > 300_000:
                        e_src_t = e_src_t[:300_000]
                    e_gen = model.transport_samples(e_src_t, t - s, n_steps=12).cpu().numpy()

                mm = extra_metrics(E_by[t], e_gen)
                rows.append(dict(target=t, source=s, delta=round(t - s, 2), method="NeuralSDE", **mm))
                print(f"  Sorgente {s} (Delta={t-s:.2f}) -> NeuralSDE: W1={mm['w1']:.4f} "
                      f"tail_err={100*mm['tail_err']:5.1f}% dose_w1={mm['dose_w1']:.4f}")

        save_rows(rows, f"{a.out}/pairs_sde.csv")

    elif a.exp == "energy":
        print("=== BENCHMARK SDE 1D DI STATO: TEST GENERALITA' ENERGIA (60 MeV) ===")
        train_d = [2.5, 3.0, 4.0]
        q_np, q_t = quantile_tables(E_by, train_d, tau_np, dev)
        model = NeuralStateSDE1D().to(dev)
        train_sde(model, q_t, tau, epochs=a.epochs, max_dd=1.5, device=dev)
        model.eval()

        tests = [(0.5, 1.0), (1.0, 2.0), (0.5, 2.0)]
        rows = []
        for s, t in tests:
            with torch.no_grad():
                e_src_t = torch.from_numpy(E_by[s]).float().to(dev)
                if len(e_src_t) > 300_000:
                    e_src_t = e_src_t[:300_000]
                e_gen = model.transport_samples(e_src_t, t - s, n_steps=12).cpu().numpy()

            mm = extra_metrics(E_by[t], e_gen)
            rows.append(dict(source=s, target=t, method="NeuralSDE", **mm))
            print(f"  Trasporto {s} -> {t} cm: W1={mm['w1']:.4f} tail_err={100*mm['tail_err']:5.1f}% "
                  f"dose_w1={mm['dose_w1']:.4f}")

        save_rows(rows, f"{a.out}/energy_sde.csv")


if __name__ == "__main__":
    main()
