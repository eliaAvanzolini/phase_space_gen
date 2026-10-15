"""depth_models.py - costruzione/caricamento/campionamento dei tre modelli (cfm | nsf | gan) a 4 canali, condizione = profondità."""
import torch


def _gen4():
    from models.gan import PhaseSpaceGenerator

    class Gen4(PhaseSpaceGenerator):
        """Generatore GAN senza la normalizzazione della direzione (canali 3:6) del progetto fotoni:
        con 4 canali (y, dx, dy, E) quella normalizzazione cadrebbe sul canale dell'energia."""
        def forward(self, z, c=None):
            x = z if c is None else torch.cat([z, c], dim=-1)
            return self.head(self.trunk(x))
    return Gen4


def build(kind, cfg, dim, dev):
    if kind == "cfm":
        from models.cfm import PhaseSpaceCFM
        m = PhaseSpaceCFM(dim=dim, cond_dim=1, hidden_dim=cfg["hidden"], n_layers=cfg["layers"])
    elif kind == "nsf":
        from models.nsf import PhaseSpaceNSF
        m = PhaseSpaceNSF(dim=dim, n_transforms=cfg["n_transforms"], hidden_dim=cfg["hidden"], n_bins=cfg["n_bins"],
                          tail_bound=cfg["tail_bound"], cond_dim=1, context_dim=cfg.get("context_dim", 32))
    elif kind == "gan":
        m = _gen4()(latent_dim=cfg["latent"], cond_dim=1, hidden_dims=cfg["gan_hidden"], output_dim=dim)
    else:
        raise ValueError(kind)
    return m.to(dev)


def load_weights(m, kind, path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m.load_state_dict(ck["generator"] if kind == "gan" else ck["model"])
    return m.eval()


@torch.no_grad()
def sample(m, kind, c, steps=200, dopri5=False):
    n = c.shape[0]
    if kind == "cfm":
        return m.sample(n, c, method="dopri5") if dopri5 else m.sample_fast(n, c, n_steps=steps)
    if kind == "nsf":
        return m.sample(n, c)
    return m.sample(n, c, device=str(c.device.type))
