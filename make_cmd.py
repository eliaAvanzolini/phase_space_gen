import json, sys
cfg = json.load(open(sys.argv[1])); seed, out, flip = sys.argv[2], sys.argv[3], sys.argv[4] == "1"
skip = {"out", "seed", "best_val", "cond_dim", "dim", "max_samples", "flip"}
parts = []
for k, v in cfg.items():
    if k in skip or v is None: continue
    if isinstance(v, bool):
        if v: parts.append(f"--{k}")
    else:
        parts.append(f"--{k} {v}")
parts += [f"--seed {seed}", f"--out {out}"]
if flip: parts.append("--flip")
print(" ".join(parts))
