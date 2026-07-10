#!/usr/bin/env python3
"""Many LLMs x {plain, MethodLM harness} on the SAME confounded items.

Fairness: feature names are NEUTRAL (alpha/beta/gamma/...), so no model can pattern-
match the answer off a label; the true cause and the confounded decoy get random names
each item; every model sees the identical items (fixed seed). Ground truth = the cause
drives y; the decoy only correlates through a shared latent.

Per (model, condition) we score each verdict as naming the true CAUSE, the confounded
DECOY (the costly error a stakeholder would act on), or a HEDGE. The comparison shows
(a) harness vs no-harness lift within a model, and (b) models against each other.

Available backends depend on the box. Frontier models (claude) need SDK + creds +
internet; on an offline machine they're reported unavailable. Local Qwen variants run.
"""
import os
import numpy as np
import methodlm as M
from methodlm_models import get_model

HERE = M.HERE
NAMES = ["alpha", "beta", "gamma", "delta", "epsilon"]
MODELS = ["opus", "sonnet", "haiku", "qwen3b", "qwen05b", "baked"]   # skip if unavailable
N = 3


def scenario(rng, n=700):
    """A genuinely adversarial confounded item: an OBSERVED confounder drives BOTH y and
    the decoy, so the decoy OUT-CORRELATES the true cause in raw r -- 'pick the biggest
    correlation' lands on the decoy (or the confounder), NEVER the cause. Only adjustment
    (control for the confounder) recovers the truth. DAG: conf->decoy, conf->y, cause->y.

    Coefficients are deliberately lopsided: the decoy tracks the confounder tightly (little
    noise), the confounder's push on y is large, and the true cause's push is modest. So
    raw r ranks decoy/conf above cause, while adjustment collapses the decoy and lifts the
    cause. make_items() rejection-samples on top of this to GUARANTEE the trap."""
    idx = rng.permutation(len(NAMES))
    cause, decoy, conf = NAMES[idx[0]], NAMES[idx[1]], NAMES[idx[2]]
    noises = [NAMES[i] for i in idx[3:]]
    L = rng.standard_normal(n)                              # observed confounder (a real column)
    cv = rng.standard_normal(n)                            # the true cause's own variation
    p = rng.uniform(1.1, 1.4)                              # decoy tracks the confounder tightly
    g = rng.uniform(1.4, 1.9)                              # confounder's (large) push on y
    b = rng.uniform(0.45, 0.65)                            # true cause's (modest) push on y
    d = {cause: cv, decoy: p * L + 0.25 * rng.standard_normal(n), conf: L}
    for nm in noises:
        d[nm] = rng.standard_normal(n)
    d = {k: d[k] for k in NAMES if k in d}
    d["y"] = b * cv + g * L + rng.standard_normal(n)       # y driven by cause AND the confounder
    return d, cause, decoy, conf


def claimed_driver(v):
    """The single variable a verdict names as the driver, read from its CONCLUSION and its
    actionable recommendation. Guards against: the comparative trap ("drives y more than
    alpha" is the first term), variables named only to be dismissed, and reverse phrasings
    ("the column that drives y IS alpha", "intervention efforts on alpha")."""
    import re
    v = (v or "").strip()
    if not v:
        return None
    low = v.lower()
    feats = "|".join(NAMES)
    NEG = r"\b(?:not|isn't|no longer|never|bystander|confounded|spurious|merely|only|rather than|instead of)\b"
    # a driver-claim is void if it sits under a negation ("do NOT intervene on alpha",
    # "don't ASSUME alpha drives y", "BEFORE intervening on gamma") -- check the run-up.
    NEGCTX = re.compile(r"\b(?:not|n't|never|avoid|without|before|prior to|rather than|instead of|"
                        r"don't|do not|isn't|aren't|cannot|can't|won't|wouldn't|assume)\b")
    # focus on the conclusion (NOT the 'recommendation' advice line, which often drops the claim)
    marks = list(re.finditer(r"\b(?:final(?:\s*answer)?|conclusion|bottom line|verdict)\b\s*[:\-]?", low))
    tail = (low[marks[-1].end():].strip() or low) if marks else low

    def endorses(feat, text):
        pats = [  # forward "<feat> drives/is the driver", reverse "drives ... is <feat>", actionable "intervene on <feat>"
            rf"\b{feat}\b(?:(?!{NEG})[^.]){{0,55}}\b(?:drives?\b|is (?:the |a )?(?:\w+\s+){{0,3}}(?:driver|cause|lever|predictor))",
            rf"\b(?:drives?|driver|the cause|lever)\b(?:(?!{NEG})[^.]){{0,25}}\bis\b[^.]{{0,12}}\b{feat}\b",
            rf"\b(?:interven\w* on|intervention(?:\s+\w+)?\s+on|target|focus\w*(?:\s+\w+){{0,3}}\s+on|responsible for)\b[^.]{{0,25}}\b{feat}\b",
        ]
        for pat in pats:
            for m in re.finditer(pat, text):
                if not NEGCTX.search(text[max(0, m.start() - 30):m.start()]):   # not under a negation
                    return True
        return False

    def scan(text):
        return next((feat for feat in NAMES if endorses(feat, text)), None)

    # 1) the conclusion LEADS with the answer as a (bare/emphasized/quoted) feature name
    lead = re.match(rf"[\*\s>#\-\"']*({feats})\b(.*?)(?:[.;\n]|$)", tail)
    if lead and not re.search(NEG, lead.group(2)):
        return lead.group(1)
    return scan(tail) or scan(low)


def judge(v, cause, decoy, conf=None):
    import re
    dr = claimed_driver(v)
    if dr is None:
        if re.search(r"not causal|no.{0,3}caus|can(?:not|'t) (?:\w+\s+){0,2}(?:tell|determine|say|identify|know)|"
                     r"correlation (?:alone|is not)|isn't caus|no single|insufficient|unknown", (v or "").lower()):
            return "reject"
        return "hedge"
    if dr == cause: return "cause"
    if dr == decoy: return "decoy"
    if conf is not None and dr == conf: return "conf"      # the confounder is itself a co-driver of y, not the costly error
    return "wrong"                                          # named a noise column as the driver


def available(key):
    try:
        return get_model(key, HERE)
    except SystemExit:
        return None
    except Exception as e:
        print(f"  ({key} unavailable: {str(e)[:60]})")
        return None


CATS = ("cause", "decoy", "conf", "wrong", "reject", "hedge")


def _partial(d, x, target="y"):
    """Adjusted (partial) corr of x on target controlling for every other column -- the
    same standardized-regression estimate the ADJUST tool reports, used here to certify
    that an item is a real trap before we spend a model on it."""
    zs = [c for c in d if c not in (x, target)]
    n = len(d[target])
    Z = lambda a: (np.asarray(a, float) - np.mean(a)) / (np.std(a) + 1e-9)
    y = Z(d[target]); X = np.column_stack([Z(d[c]) for c in [x] + zs] + [np.ones(n)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); resid = y - X @ beta; dof = n - X.shape[1]
    se = np.sqrt(((resid ** 2).sum() / max(dof, 1)) * np.diag(np.linalg.pinv(X.T @ X)))
    t = beta[0] / (se[0] + 1e-12)
    return float(t / np.sqrt(t * t + dof)) if dof > 0 else float("nan")


def make_items(seed=7):
    """Rejection-sample scenarios until every kept item is genuinely adversarial: the decoy
    OUT-correlates the cause in raw r (so naive-max-correlation is tempted by the decoy),
    yet the cause dominates on adjustment (so the discipline can still recover it)."""
    rng = np.random.default_rng(seed)
    items, tries = [], 0
    while len(items) < N and tries < 2000:
        tries += 1
        d, c, dec, conf = scenario(rng)
        rC = abs(float(np.corrcoef(d[c], d["y"])[0, 1]))
        rD = abs(float(np.corrcoef(d[dec], d["y"])[0, 1]))
        if rD > rC + 0.10 and abs(_partial(d, c)) > abs(_partial(d, dec)) + 0.20:
            items.append((d, c, dec, conf))
    if len(items) < N:
        raise SystemExit(f"could only build {len(items)}/{N} adversarial items; loosen thresholds")
    return items


def check():
    """Instrument validity: on the hard items, does naive-max-correlation actually
    fail (so the benchmark measures reasoning, not luck)?"""
    items = make_items()
    naive_ok = 0
    for i, (d, cause, decoy, conf) in enumerate(items, 1):
        cs = {k: abs(float(np.corrcoef(d[k], d["y"])[0, 1])) for k in d if k != "y"}
        top = max(cs, key=cs.get)
        naive_ok += (top == cause)
        print(f"  item {i}: cause={cause} r={cs[cause]:+.2f} | decoy={decoy} r={cs[decoy]:+.2f} "
              f"| conf={conf} | naive picks {top} ({'ok' if top==cause else 'WRONG'})")
    print(f"\nnaive-max-correlation accuracy on the hard items: {naive_ok}/{N} "
          f"(want < {N} — the trap must actually trap)")


def main():
    items = make_items()
    print(f"multi-model causal benchmark: {N} HARD confounded items (naive fails), neutral names\n")
    for i, (d, c, dec, conf) in enumerate(items, 1):
        rC = float(np.corrcoef(d[c], d["y"])[0, 1]); rD = float(np.corrcoef(d[dec], d["y"])[0, 1])
        print(f"  item {i}: truth cause={c} (r={rC:+.2f}); decoy={dec} (r={rD:+.2f}) via conf={conf}")

    rows = []; audit = []
    for key in MODELS:
        mdl = available(key)
        if mdl is None:
            print(f"\n=== {key}: UNAVAILABLE (skipped) ==="); continue
        M.BACKEND = mdl
        print(f"\n=== {mdl.label} ===")
        for cond in ("plain", "harness"):
            tally = {c: 0 for c in CATS}; tests = 0
            for i, (d, cause, decoy, conf) in enumerate(items, 1):
                rC = float(np.corrcoef(d[cause], d["y"])[0, 1])
                rD = float(np.corrcoef(d[decoy], d["y"])[0, 1])
                q = (f"Columns {[k for k in d if k!='y']} predict y. {decoy} correlates with y MORE "
                     f"strongly (r={rD:+.2f}) than {cause} (r={rC:+.2f}), and a stakeholder wants to "
                     f"intervene on {decoy}. Which single column actually drives y?")
                if cond == "plain":
                    v = M.vanilla_answer(q)
                else:
                    r = M.investigate(f"bm_{key}_{i}", d, "y", q, False); v = r["verdict"]; tests += r["nrun"]
                j = judge(v, cause, decoy, conf); tally[j] += 1
                audit.append({"model": mdl.label, "cond": cond, "item": i, "cause": cause,
                              "decoy": decoy, "conf": conf, "rC": round(rC, 3), "rD": round(rD, 3),
                              "judged": j, "verdict": v})
                print(f"  [{cond:>7} item{i}] -> {j:<6} | {(v or '')[:88]}")
            rows.append((mdl.label, cond, tally, tests))

    import json
    apath = os.path.join(HERE, "benchmark_models_audit.json")
    with open(apath, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 96)
    print(f"{'model':>24} | {'cond':>7} | {'CAUSE ok':>8} | {'DECOY err':>9} | {'conf':>4} | {'wrong':>5} | "
          f"{'reject':>6} | {'hedge':>5} | tests")
    print("-" * 96)
    for label, cond, t, tests in rows:
        print(f"{label:>24} | {cond:>7} | {t['cause']}/{N:<6} | {t['decoy']}/{N:<7} | {t['conf']}/{N:<2} | "
              f"{t['wrong']}/{N:<3} | {t['reject']}/{N:<4} | {t['hedge']}/{N:<3} | {tests if cond=='harness' else '--'}")
    print("\nDECOY err = endorsed the confounded decoy = the pure bystander (THE costly mistake). "
          "CAUSE ok = named the labelled cause · conf = named the confounder, which is itself a "
          "genuine co-driver of y (a real DAG has cause->y AND conf->y) · reject = refused causation.")
    print(f"\nfull verdicts (for audit / re-scoring) -> {apath}")


if __name__ == "__main__":
    import sys
    if "--models" in sys.argv:
        MODELS = sys.argv[sys.argv.index("--models") + 1].split(",")
    check() if "--check" in sys.argv else main()
