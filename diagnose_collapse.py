"""
Collasso di sigma(dx), sigma(dy) in funzione di E, v=E^1.77, u=v+k*d.
Metrica: errore LOO su log sigma (|log s_d - media log s delle altre profondita' di training|).
3.5 cm e' trattato come test (mai usato per la media).
"""
import numpy as np
from pathlib import Path
import read_topas_phsp as rtp

TRAIN = [0.5, 1.0, 2.0, 2.5, 3.0, 4.0]
K, P, MIN_N, NB = 450.0, 1.77, 3000, 40


def load(path):
    df = rtp.read_phsp(str(path))
    if "pdg" in df.columns:
        df = df[df["pdg"] == 2212]
    ps = rtp.to_7d(df).astype(np.float64)
    ps = ps[ps[:, 6] > 0]
    return ps[:, 6], ps[:, 3], ps[:, 4]


def rsd(a):
    q1, q3 = np.percentile(a, [25, 75])
    return (q3 - q1) / 1.349


data = {}
for p in sorted(Path("data/raw_phsp").glob("*peak*.phsp")):
    d = rtp.parse_depth_from_name(p.name)
    if d is None:
        continue
    d = round(float(d), 3)
    if d in TRAIN or d == 3.5:
        data[d] = load(p)
        print("caricato", d, len(data[d][0]))

coords = {
    "E": lambda E, d: E,
    "v=E^1.77": lambda E, d: E ** P,
    "u=v+k*d": lambda E, d: E ** P + K * d,
}

for cname, f in coords.items():
    cv = {d: f(data[d][0], d) for d in data}
    pooled = np.concatenate([cv[d] for d in data])
    edges = np.linspace(np.percentile(pooled, 0.5), np.percentile(pooled, 99.5), NB + 1)
    for vi, vname in ((1, "dx"), (2, "dy")):
        sig = {}
        for d in data:
            s = np.full(NB, np.nan)
            idx = np.digitize(cv[d], edges) - 1
            for b in range(NB):
                m = idx == b
                if m.sum() >= MIN_N:
                    s[b] = np.log(rsd(data[d][vi][m]))
            sig[d] = s
        res = {}
        for d in data:
            others = [sig[o] for o in TRAIN if o != d]
            stack = np.vstack(others)
            ok = (~np.isnan(sig[d])) & ((~np.isnan(stack)).sum(0) >= 2)
            if ok.sum() == 0:
                res[d] = (np.nan, 0)
                continue
            pred = np.nanmean(stack[:, ok], axis=0)
            res[d] = (float(np.mean(np.abs(sig[d][ok] - pred))), int(ok.sum()))
        tr_vals = [res[d][0] for d in TRAIN if not np.isnan(res[d][0])]
        line = " ".join(f"{d}:{res[d][0]*100:5.1f}%({res[d][1]})" for d in sorted(res))
        print(f"{cname:10s} {vname} | LOO medio train = {100*np.mean(tr_vals):5.1f}% | {line}")
