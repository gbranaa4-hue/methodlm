"""Re-score a saved benchmark audit JSON with the CURRENT judge (no API spend)."""
import json, sys
from collections import Counter
from benchmark_models import judge

path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_models_audit.json"
d = json.load(open(path, encoding="utf-8"))
models = dict.fromkeys(r["model"] for r in d)
for model in models:
    for cond in ("plain", "harness"):
        rows = [r for r in d if r["model"] == model and r["cond"] == cond]
        if not rows:
            continue
        t = Counter(judge(r["verdict"], r["cause"], r["decoy"], r["conf"]) for r in rows)
        n = len(rows)
        print(f"{model:>22} | {cond:>7} | CAUSE {t['cause']}/{n} | DECOY {t['decoy']}/{n} | "
              f"conf {t['conf']}/{n} | wrong {t['wrong']}/{n} | reject {t['reject']}/{n} | hedge {t['hedge']}/{n}")
