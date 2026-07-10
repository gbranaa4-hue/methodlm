"""Instrument test: does the FIXED judge score the real verdicts from the paid frontier
run correctly? Verdict strings are transcribed verbatim from the run transcript. This is a
pre-registered check -- expected labels are written down, then asserted, so a regression in
the judge is caught rather than silently mis-scoring a published table."""
from benchmark_models import claimed_driver, judge

# (verdict, cause, decoy, expected_label). Harness verdicts are full text from the traces.
CASES = [
    # --- Opus 4.8, harness ---
    ("gamma drives y. It survived adjustment with a large, robust partial correlation "
     "(+0.71, RV=0.62), while alpha collapsed to +0.04 (RV=0.04), marking it a confounded "
     "bystander. Advise the stakeholder against intervening on alpha.", "gamma", "alpha", "cause"),
    ("alpha drives y. Its adjusted effect held (partial corr +0.67, RV=0.58) while gamma "
     "vanished under adjustment (+0.03, RV=0.03), marking gamma as a confounded bystander. "
     "Intervening on gamma would waste effort; target alpha.", "alpha", "gamma", "cause"),
    # --- Sonnet 5, harness: "is the ACTUAL driver" (adjective gap) + "alpha is not causal" (negation) ---
    ("gamma is the actual driver of y. alpha is not causal - intervening on alpha will not "
     "move y; the stakeholder should target gamma instead.", "gamma", "alpha", "cause"),
    ("gamma is the driver. Its partial correlation with y survives adjustment for alpha, "
     "beta, delta, epsilon (actually strengthens to +0.71, RV=0.62). Alpha's correlation "
     "collapsed to ~0 (RV=0.04), indicating alpha is a confounded bystander, not a cause.",
     "gamma", "alpha", "cause"),
    ("alpha is the driver of y. Gamma is a confounded bystander (its effect vanishes once "
     "alpha and other covariates are controlled).", "alpha", "gamma", "cause"),
    # --- Haiku 4.5, harness: terse one-word FINAL ---
    ("gamma", "gamma", "alpha", "cause"),
    ("gamma drives y. Alpha's raw correlation was spurious confounding; it collapsed to "
     "r=+0.01 under adjustment (RV=0.01). Gamma survived and strengthened (r=+0.67).",
     "gamma", "alpha", "cause"),
    # --- the comparative-phrasing trap that USED to mis-score as the decoy ---
    ("# Conclusion **Gamma drives y more than alpha.** The correlation of r=+0.64 is "
     "meaningful.", "gamma", "alpha", "cause"),
    ("Based on the correlation data alone, **alpha appears to drive y more than gamma**.",
     "alpha", "gamma", "cause"),
    # --- honest refusals should score reject, not a driver ---
    ("## Conclusion **Correlation cannot tell you which column drives y.** Both alpha "
     "(r=+0.45) and gamma look similar.", "gamma", "alpha", "reject"),
    # --- a genuine decoy endorsement must still be caught ---
    ("Gamma drives y; the stakeholder should intervene on gamma.", "alpha", "gamma", "decoy"),
    # --- naming a control/bystander as the driver must be caught as wrong ---
    ("epsilon is the true driver of y.", "gamma", "alpha", "wrong"),
    # --- reverse phrasing: the real Qwen-3B plain verdicts (were mis-scored as hedge) ---
    ("The single column that drives y is alpha, as it has the highest correlation with y "
     "(r=+0.75) compared to gamma (r=+0.36).", "gamma", "alpha", "decoy"),
    ("Conclusion: Column 'alpha' is the column that most strongly drives the prediction of "
     "y (r=+0.78).\n\nRecommendation: Focus intervention efforts on column 'alpha'.",
     "gamma", "alpha", "decoy"),
    # --- careful refusals with a NEGATED action must score reject, not decoy (the Opus/Haiku plain bug) ---
    ("Conclusion: Correlation cannot tell you which column drives y. Do not intervene on "
     "alpha based on correlation alone; run a controlled experiment first.", "gamma", "alpha", "reject"),
    ("Correlation cannot tell you which column drives y. Do not assume alpha drives y just "
     "because r is highest; the true driver is currently unknown.", "gamma", "alpha", "reject"),
    ("I cannot definitively tell you which column drives y. Before the stakeholder intervenes "
     "on alpha, run a multiple regression to check confounding.", "gamma", "alpha", "reject"),
    # --- a real harness slip: adjusting for the WRONG variable, so the decoy 'survives' (Sonnet item2) ---
    ("alpha is the driver of y. Its adjusted effect controlling for gamma stayed strong "
     "(+0.81) with high robustness (RV=0.72) - the stakeholder's target is justified.",
     "gamma", "alpha", "decoy"),
]

ok = 0
for v, cause, decoy, want in CASES:
    got = judge(v, cause, decoy)
    flag = "OK " if got == want else "XX "
    ok += got == want
    print(f"  {flag} want={want:<6} got={got:<6} | {v[:70]}")
print(f"\n{ok}/{len(CASES)} judge cases correct")
assert ok == len(CASES), "judge regressed on a pre-registered case"
print("instrument OK -- judge scores the real verdicts as intended")
