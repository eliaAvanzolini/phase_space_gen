"""
train_depth_gan.py - WGAN-GP condizionato sulla profondità (models/gan.py del progetto fotoni, generatore a 4 canali).

    python -u train_depth_gan.py --data data/proton_depth_train.h5 --out runs/gan_R2 --epochs 200

Il GAN non ha una loss di validazione: si addestra per --epochs epoche (default 200, come il CFM), si salva un checkpoint
ogni --save_every_epochs (ckpt_ep<N>.pt) e alla fine last.pt. Eventuale scelta del checkpoint: guardare eval_depth.py
sulle profondità di training (NON sull'hold-out). Iperparametri di default = ESATTAMENTE config gan_conditional dei fotoni
(hidden [400,400,400], n_critic 4, batch 4096, lr 1e-5, lambda_gp 10).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from depth_data import add_data_args, load_standardized
from depth_models import _gen4
from phsp_transforms import save_stats


def main():
    ap = argparse.ArgumentParser(); add_data_args(ap)
    ap.add_argument("--epochs", type=int, default=200, help="1 epoca = tanti passi del generatore quanti servono a vedere N campioni (N/batch)")
    ap.add_argument("--steps", type=int, default=None, help="alternativa: numero totale di passi del generatore (ignora --epochs)")
    ap.add_argument("--batch", type=int, default=4096); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n_critic", type=int, default=4); ap.add_argument("--lambda_gp", type=float, default=10.0)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--gan_hidden", type=int, nargs="+", default=[400, 400, 400])
    ap.add_argument("--save_every_epochs", type=int, default=20)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    X, C, stats = load_standardized(a, rng); save_stats(stats, out / "stats.json")
    Xt, Ct = torch.from_numpy(X).to(dev), torch.from_numpy(C).to(dev); N, dim = len(Xt), Xt.shape[1]

    from models.gan import PhaseSpaceCritic, WGANGPTrainer
    G = _gen4()(latent_dim=a.latent, cond_dim=1, hidden_dims=a.gan_hidden, output_dim=dim)
    D = PhaseSpaceCritic(input_dim=dim, cond_dim=1, hidden_dims=a.gan_hidden)
    tr_ = WGANGPTrainer(G, D, device=dev, lr=a.lr, n_critic=a.n_critic, lambda_gp=a.lambda_gp)
    print(f"G: {sum(p.numel() for p in G.parameters())/1e6:.2f} M, D: {sum(p.numel() for p in D.parameters())/1e6:.2f} M, device={dev}")

    spe = max(1, N // a.batch)                              # passi per epoca
    total = a.steps if a.steps else a.epochs * spe
    print(f"passi per epoca: {spe}, passi totali del generatore: {total}")
    t0 = time.time(); acc = []
    for step in range(1, total + 1):
        idx = torch.randint(0, N, (a.batch,), device=dev)
        m = tr_.train_step(Xt[idx], Ct[idx]); acc.append([m["loss_D"], m["loss_G"], m["w_dist"], m["gp"]])
        if step % spe == 0:
            ep = step // spe; v = np.mean(acc, 0); acc = []
            print(f"ep {ep:4d}  loss_D {v[0]:+.3f}  loss_G {v[1]:+.3f}  W {v[2]:.4f}  gp {v[3]:.3f}  [{time.time()-t0:.0f}s]", flush=True)
            if ep % a.save_every_epochs == 0:
                tr_.save(str(out / f"ckpt_ep{ep}.pt"))
    tr_.save(str(out / "last.pt"))
    json.dump(dict(vars(a), model="gan"), open(out / "run.json", "w"), indent=2)


if __name__ == "__main__":
    main()
