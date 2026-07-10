#!/usr/bin/env python3
"""Causal-reasoning benchmark: does the method reject confounded decoys?

Each scenario has a KNOWN ground truth (a confounding DAG):
    Sea -> cause, Sea -> decoy, cause -> outcome      (+ pure-noise features)
So `cause` truly drives `outcome`; `decoy` only correlates with it through the shared
latent `Sea` (the humidity/temperature structure, generalized). Ground-truth causal
set = {cause}.

Two statistical arms, scored over K scenarios:
  naive   : flag a feature causal if |corr(feature, outcome)| > tau
  method  : flag it if |partial corr controlling for the other features| > tau
            (= MethodLM's ADJUST/gate logic: backdoor adjustment)

The metric that matters is the FALSE-POSITIVE RATE ON THE DECOY -- how often each arm
calls a confounded bystander a cause (the "temperature causes error, buy cooling"
mistake). Pre-registered: naive flags the decoy most of the time; method rarely does,
while both keep a high true-positive rate on the real cause.

Optional --llm arm: on a few scenarios, run a plain LLM vs the full MethodLM harness
and check which one refuses the decoy in words.

Reproducible: fixed seed. This is SYNTHETIC (known ground truth) and mirrors the
structure of public causal-reasoning suites (Corr2Cause / CLADDER); running on those
is the online next step.
"""
import argparse
import numpy as np

TAU = 0.10
rng = np.random.default_rng(20260709)


def scenario(n=600, n_noise=3):
    sea = rng.standard_normal(n)
    a, c, b = rng.uniform(0.6, 0.9), rng.uniform(0.6, 0.9), rng.uniform(0.7, 1.0)
    cause = a * sea + rng.standard_normal(n)
    decoy = c * sea + rng.standard_normal(n)
    outcome = b * cause + rng.standard_normal(n)          # outcome driven by cause ONLY
    d = {"cause": cause, "decoy": decoy}
    for i in range(n_noise):
        d[f"noise{i+1}"] = rng.standard_normal(n)
    d["outcome"] = outcome
    return d


def partial_all(d, x, target="outcome"):
    Z = [c for c in d if c not in (x, target)]
    n = len(d[target])
    A = np.column_stack([d[c] for c in Z] + [np.ones(n)])
    def resid(v):
        beta, *_ = np.linalg.lstsq(A, v, rcond=None); return v - A @ beta
    return float(np.corrcoef(resid(d[x]), resid(d[target]))[0, 1])


def stat_benchmark(K=120):
    feats = None
    tally = {"naive": {}, "method": {}}
    val = {"naive": {"cause": [], "decoy": []}, "method": {"cause": [], "decoy": []}}
    for _ in range(K):
        d = scenario()
        feats = [c for c in d if c != "outcome"]
        for f in feats:
            raw = abs(float(np.corrcoef(d[f], d["outcome"])[0, 1]))
            par = abs(partial_all(d, f))
            for arm, s in (("naive", raw), ("method", par)):
                tally[arm].setdefault(f, 0)
                tally[arm][f] += int(s > TAU)
            role = f if f in ("cause", "decoy") else "noise"
            if role in ("cause", "decoy"):
                val["naive"][role].append(raw); val["method"][role].append(par)
    noise = [f for f in feats if f.startswith("noise")]
    def rate(arm, f): return tally[arm][f] / K
    def noise_rate(arm): return np.mean([tally[arm][f] for f in noise]) / K

    print(f"causal benchmark: {K} confounded scenarios, tau={TAU}\n")
    print(f"{'arm':>8} | {'flags CAUSE':>11} | {'flags DECOY':>11} | {'flags noise':>11} | mean |assoc|")
    print("-" * 74)
    for arm in ("naive", "method"):
        mc = np.mean(val[arm]["cause"]); md = np.mean(val[arm]["decoy"])
        print(f"{arm:>8} | {rate(arm,'cause')*100:>10.0f}% | {rate(arm,'decoy')*100:>10.0f}% | "
              f"{noise_rate(arm)*100:>10.0f}% | cause {mc:+.2f} · decoy {md:+.2f}")
    dn, dm = rate("naive", "decoy"), rate("method", "decoy")
    print(f"\nHEADLINE: naive correlation calls the confounded decoy a cause {dn*100:.0f}% of the "
          f"time;\n          the method (backdoor adjustment) does so {dm*100:.0f}% of the time"
          + (f" -- a {(dn-dm)/max(dn,1e-9)*100:.0f}% cut in false causal claims." if dn > dm else "."))
    tp = rate("method", "cause")
    print(f"          Method keeps {tp*100:.0f}% true-positive on the real cause.")
    return {"naive_decoy_fp": dn, "method_decoy_fp": dm, "method_cause_tp": tp}


def llm_arm(n_items=2):
    import methodlm as M
    M.BACKEND = M.methodlm_models.get_model("local", M.HERE)
    print(f"\n--- LLM arm: plain model vs MethodLM harness ({M.BACKEND.label}) ---")
    for i in range(n_items):
        d = scenario()
        rC = float(np.corrcoef(d["cause"], d["outcome"])[0, 1])
        rD = float(np.corrcoef(d["decoy"], d["outcome"])[0, 1])
        q = (f"Our outcome correlates with decoy (r={rD:+.2f}) and with cause (r={rC:+.2f}). "
             "A stakeholder wants to intervene on decoy. What actually drives outcome?")
        print(f"\n[item {i+1}] truth: cause drives outcome; decoy is a confounded bystander "
              f"(corr decoy {rD:+.2f}, cause {rC:+.2f})")
        van = M.vanilla_answer(q)
        print(f"  plain LLM   : {van[:180]}")
        res = M.investigate(f"bench{i+1}", d, "outcome", q, False)
        v = res["verdict"]
        endorses_decoy = "decoy" in v.lower() and "not" not in v.lower()[:v.lower().find("decoy")+6]
        print(f"  MethodLM    : ({res['nrun']} test) {v[:180]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also run the slow LLM arm")
    ap.add_argument("-K", type=int, default=120)
    args = ap.parse_args()
    stat_benchmark(args.K)
    if args.llm:
        llm_arm()
