"""
phsp_transforms.py  (numpy + scipy, niente torch)
=================================================
Trasformazione 7D (x,y,z,dx,dy,dz,E)  <->  4 canali modellati  (y, dx, dy, log E)  standardizzati.

Scelte, motivate dai dati (file 3.5 e inventario):
  * x NON è modellata: uniforme in [-0.05, 0.05] cm (geometria dello scoring), indipendente dal resto.
  * z costante per profondità: z = -(Z0 + depth).
  * dz non è modellata: negativa nel 99.998% dei protoni -> dz = -sqrt(1 - dx^2 - dy^2).
  * E -> log E (garantisce E > 0).

STANDARDIZZAZIONE PER PROFONDITÀ (mode="depth", default)
  Ogni canale è centrato e scalato con media/std DELLA PROPRIA profondità. Il flow impara solo la forma
  residua (che varia poco con d), mentre posizione e larghezza (che variano tanto, soprattutto E) sono
  gestite da una curva mu(d), sigma(d) interpolata tra le profondità di training.
  Interpolazione (interp):
    "hybrid"   : posizione come pchip_Ep, scala log-lineare (consigliato con etrans="lin")
    "pchip_Ep" : PCHIP su  exp(p*mean logE)  (~ E^p, p=1.77 Bragg-Kleeman: quasi lineare in d) per logE,
                 PCHIP su media per gli altri canali, PCHIP su log(std) per tutte le std.   [default]
    "pchip"    : PCHIP su media e log(std) di tutti i canali
    "linear"   : interpolazione lineare
  stat ("moments" | "robust"): media/std oppure mediana/(IQR/1.349) come posizione/scala per ogni canale.
  mode="global": vecchio comportamento (una sola media/std per canale su tutto il training).
"""
from __future__ import annotations
import json
import numpy as np
from scipy.interpolate import PchipInterpolator

CHANNELS = ["y", "dx", "dy", "logE"]
Z0_CM = 158.0
P_BRAGG = 1.77


# ───────────────────────── trasformazione di E ────────────────────────────────
# Il 4° canale è f(E): "log" (default storico), "lin" (E in MeV) oppure "sqrt".
# La scelta conta perché la scala di f(E) deve variare in modo liscio con d per poter essere interpolata.
def e_fwd(E, etrans="log", p=P_BRAGG):
    E = np.maximum(np.asarray(E, np.float64), 1e-6)
    if etrans == "log":  return np.log(E)
    if etrans == "sqrt": return np.sqrt(E)
    if etrans == "Ep":   return E ** p            # v = E^p: la profondità è ~ una traslazione in v
    return E


def e_inv(v, etrans="log", p=P_BRAGG):
    v = np.asarray(v, np.float64)
    if etrans == "log":
        return np.exp(v)
    if etrans == "Ep":                              # v < 0 (non fisico, bordo di arresto): riflesso
        return np.maximum(np.abs(v), 1e-9) ** (1.0 / p)
    E = v ** 2 if etrans == "sqrt" else v
    return np.maximum(np.abs(E) if etrans == "lin" else E, 1e-4)   # lin: riflette i valori negativi


def loc_to_E(loc, etrans="log", p=P_BRAGG):
    """posizione (media/mediana) del canale f(E) -> energia corrispondente [MeV], senza clipping."""
    loc = np.asarray(loc, np.float64)
    if etrans == "log":  return np.exp(loc)
    if etrans == "sqrt": return loc ** 2
    if etrans == "Ep":   return np.maximum(loc, 0.0) ** (1.0 / p)
    return loc


# ───────────────────────── 7D <-> feature ─────────────────────────────────────

def to_features(ps7: np.ndarray, etrans: str = "log", p: float = P_BRAGG) -> np.ndarray:
    """(N,7) -> (N,4) = (y, dx, dy, f(E))"""
    return np.column_stack([ps7[:, 1], ps7[:, 3], ps7[:, 4], e_fwd(ps7[:, 6], etrans, p)]).astype(np.float32)


def features_to_7d(feat: np.ndarray, depth: float, stats: dict, rng=None) -> np.ndarray:
    """(N,4) in unità fisiche -> (N,7) = (x,y,z,dx,dy,dz,E). x campionata uniforme."""
    rng = rng or np.random.default_rng()
    n = len(feat)
    y, dx, dy, fE = feat.T
    r2 = dx ** 2 + dy ** 2
    bad = r2 >= 0.999999                       # campioni non fisici (rarissimi): riscalati sul cerchio unitario
    if bad.any():
        s = np.sqrt(0.999999 / r2[bad]); dx = dx.copy(); dy = dy.copy()
        dx[bad] *= s; dy[bad] *= s
    dz = -np.sqrt(1.0 - dx ** 2 - dy ** 2)
    x = rng.uniform(-stats["x_halfwidth"], stats["x_halfwidth"], n)
    z = np.full(n, -(stats.get("z0", Z0_CM) + depth))
    return np.column_stack([x, y, z, dx, dy, dz, e_inv(fE, stats.get("etrans", "log"), stats.get("p", P_BRAGG))]).astype(np.float32)


# ───────────────────────── statistiche per profondità ─────────────────────────

def loc_scale(x: np.ndarray, stat: str = "moments"):
    """Posizione e scala per canale. moments: media/std. robust: mediana/(IQR/1.349) (insensibile alle code)."""
    if stat == "robust":
        q25, med, q75 = np.percentile(x, [25, 50, 75], axis=0)
        return med.astype(np.float64), ((q75 - q25) / 1.349).astype(np.float64)
    return x.mean(0, dtype=np.float64), x.std(0, dtype=np.float64)


def fit_depth_stats(feat: np.ndarray, depths: np.ndarray, x_halfwidth: float = 0.05,
                    mode: str = "depth", interp: str = "pchip_Ep", p: float = P_BRAGG,
                    stat: str = "moments", etrans: str = "log",
                    translate: bool = False, fit_dmax: float = 3.0, asinh: bool = False) -> dict:
    """Posizione/scala per canale, per ogni profondità presente (+ media/std globali).

    translate=True (pensato per etrans="Ep"): il canale di E NON usa la tabella per profondità ma
        posizione(d) = a - k*d   (retta fittata sulle MEDIANE delle profondità <= fit_dmax, dove il picco
                                  principale non è ancora tagliato dall'arresto dei protoni)
        scala        = s costante (IQR/1.349 dei residui, raggruppati)
    asinh=True: sul canale di E si applica arcsinh a z=(v-mu)/s: lascia il picco principale (|z|<~1) praticamente
        intatto e comprime la coda fino a v=0 (che altrimenti sta a ~-45 sigma dal picco) a pochi sigma.
    Ipotesi verificata con plot_Ep_alignment.py: gli spettri in E^p sono lo stesso spettro traslato."""
    depths = np.asarray(depths, np.float64)
    ud = np.unique(np.round(depths, 6))
    LS = [loc_scale(feat[np.isclose(depths, d, atol=1e-6)], stat) for d in ud]
    M = np.stack([a for a, _ in LS]); S = np.stack([b for _, b in LS])
    tr = None
    if translate:
        sel = [i for i, d in enumerate(ud) if d <= fit_dmax + 1e-9]
        if len(sel) < 2:
            raise ValueError("translate: servono almeno 2 profondità <= fit_dmax")
        med = np.array([np.median(feat[np.isclose(depths, ud[i], atol=1e-6), -1]) for i in sel])
        slope, a0 = np.polyfit(ud[sel], med, 1)
        res = np.concatenate([feat[np.isclose(depths, ud[i], atol=1e-6), -1] - (a0 + slope * ud[i]) for i in sel])
        q25, q75 = np.percentile(res, [25, 75])
        tr = dict(a=float(a0), k=float(-slope), s=float((q75 - q25) / 1.349), fit_dmax=float(fit_dmax), asinh=bool(asinh))
    return dict(
        translate=tr,
        mode=mode, interp=interp, p=p, stat=stat, etrans=etrans, channels=CHANNELS, z0=Z0_CM, x_halfwidth=float(x_halfwidth),
        depths=ud.tolist(), mean_k=M.tolist(), std_k=S.tolist(),
        mean=feat.mean(0, dtype=np.float64).tolist(), std=feat.std(0, dtype=np.float64).tolist(),
        d_min=float(ud.min()), d_max=float(ud.max()),
    )


def interp_from_table(ds, M, S, d, p=P_BRAGG, interp="pchip_Ep", channels=CHANNELS, etrans="log"):
    """Posizione e scala (4,) a profondità d, dalla tabella (ds, M, S) delle profondità note.

    interp:
      "hybrid"   : posizione = pchip_Ep (E: PCHIP su E^p, altri canali: PCHIP), scala = log-lineare   [consigliato con etrans=lin]
      "pchip_Ep" : come hybrid ma scala = PCHIP su log(scala)
      "pchip"    : PCHIP su posizione e log(scala), tutti i canali
      "linear"   : lineare su posizione, log-lineare su scala
    """
    ds, M, S = np.asarray(ds, float), np.asarray(M, float), np.asarray(S, float)
    hit = np.where(np.isclose(ds, d, atol=1e-6))[0]
    if len(hit):                                    # profondità di training: valori esatti
        return M[hit[0]].copy(), S[hit[0]].copy()
    if len(ds) < 2:
        raise ValueError("servono almeno 2 profondità per interpolare")
    n = M.shape[1]; je = n - 1                      # canale di E = ultimo
    mean, std = np.empty(n), np.empty(n)
    for j in range(n):
        # posizione
        if interp == "linear":
            mean[j] = np.interp(d, ds, M[:, j])
        elif j == je and interp in ("pchip_Ep", "hybrid"):
            u_k = np.maximum(loc_to_E(M[:, j], etrans, p), 1e-9) ** p          # ~ E^p: quasi lineare in d
            u = float(PchipInterpolator(ds, u_k, extrapolate=True)(d))
            mean[j] = float(e_fwd(max(u, 1e-12) ** (1.0 / p), etrans, p))
        else:
            mean[j] = float(PchipInterpolator(ds, M[:, j], extrapolate=True)(d))
        # scala
        if interp in ("linear", "hybrid"):
            std[j] = float(np.exp(np.interp(d, ds, np.log(S[:, j]))))
        else:
            std[j] = float(np.exp(PchipInterpolator(ds, np.log(S[:, j]), extrapolate=True)(d)))
    return mean, std


def stats_at(stats: dict, d: float, interp: str | None = None):
    """(mean(4), std(4)) da usare per standardizzare/destandardizzare alla profondità d."""
    if stats.get("mode", "global") == "global":
        return np.asarray(stats["mean"]), np.asarray(stats["std"])
    mean, std = interp_from_table(stats["depths"], stats["mean_k"], stats["std_k"], d,
                                  stats.get("p", P_BRAGG), interp or stats.get("interp", "pchip_Ep"),
                                  stats.get("channels", CHANNELS), stats.get("etrans", "log"))
    tr = stats.get("translate")
    if tr:                                           # E: retta in d + scala costante (niente interpolazione di tabella)
        mean = np.array(mean, float); std = np.array(std, float)
        mean[-1] = tr["a"] - tr["k"] * float(d); std[-1] = tr["s"]
    return mean, std


def standardize_for_training(feat: np.ndarray, depths: np.ndarray, stats: dict) -> np.ndarray:
    """Standardizza ogni campione con le statistiche della sua profondità (training: valori esatti)."""
    out = np.empty_like(feat, dtype=np.float32)
    depths = np.asarray(depths, np.float64)
    for d in np.unique(np.round(depths, 6)):
        m = np.isclose(depths, d, atol=1e-6)
        mu, sd = stats_at(stats, float(d))
        out[m] = ((feat[m] - mu) / sd).astype(np.float32)
    if (stats.get("translate") or {}).get("asinh"):
        out[:, -1] = np.arcsinh(out[:, -1])
    return out


def destandardize_at(z: np.ndarray, depth: float, stats: dict) -> np.ndarray:
    """z standardizzato -> unità fisiche, con mu(d), sigma(d) interpolati a `depth`."""
    mu, sd = stats_at(stats, float(depth))
    if (stats.get("translate") or {}).get("asinh"):
        z = np.array(z, dtype=np.float64, copy=True); z[:, -1] = np.sinh(z[:, -1])
    return (z * sd + mu).astype(np.float32)


# ───────────────────────── condizione e I/O ───────────────────────────────────

def norm_depth(d, stats):
    """profondità [cm] -> [0,1] (fuori da [d_min,d_max] = estrapolazione)."""
    return ((np.asarray(d, np.float32) - stats["d_min"]) / (stats["d_max"] - stats["d_min"])).astype(np.float32)


def save_stats(stats, path):
    json.dump(stats, open(path, "w"), indent=2)


def load_stats(path):
    return json.load(open(path))
