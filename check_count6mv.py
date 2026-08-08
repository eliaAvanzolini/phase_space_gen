import glob
import uproot

CLASSES = ["6mv_5x5", "6mv_10x10", "6mv_20x20"]
BASE = "outputs/gate_jaw_ref_6mv_part2_fixed"

# Riferimento per il confronto (part1/part3/part4, gia' noti e mutuamente coerenti)
REFERENCE_COUNTS = {
    "6mv_5x5":   {"part1": 48397, "part3": 48241, "part4": 48554},
    "6mv_10x10": {"part1": 111575, "part3": 111120, "part4": 111690},
    "6mv_20x20": {"part1": 383069, "part3": 382633, "part4": 383721},
}

for cls in CLASSES:
    files = sorted(glob.glob(f"{BASE}/{cls}/{cls}_phsp_part*.root"))
    total = 0
    for f in files:
        with uproot.open(f) as fh:
            keys = fh.keys()
            if not keys:
                continue
            total += fh[keys[0]].num_entries

    ref = REFERENCE_COUNTS[cls]
    avg_ref = sum(ref.values()) / len(ref)
    ratio = total / avg_ref if avg_ref > 0 else float("nan")
    flag = "✅ coerente" if 0.5 < ratio < 2.0 else "⚠️ ANCORA ANOMALO"

    print(f"{cls}: {len(files)} file, {total:,} particelle totali "
          f"(part1/3/4 medio: {avg_ref:,.0f}, rapporto: {ratio:.2f}x) {flag}")
