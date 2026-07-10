"""Instrument test for the ADJUST collider/mediator bias audit.

Synthetic DAG (X = exposure, Y = target):
  conf  -> X, conf -> Y          (confounder: a prior common cause; adjusting is CORRECT)
  X -> med -> Y                   (mediator: on the causal path; adjusting is WRONG)
  X -> coll <- Y                  (collider: a common effect; adjusting is WRONG, opens a path)
  noise                          (neutral: unrelated)

Pre-registered: the audit must flag `coll` as a COLLIDER (data-detectable), must NOT flag
`noise`, and must surface `med`/`conf` as explain-away (ambiguous -> defer to the DAG). It
can never distinguish `med` from `conf` by data alone; that's the honest boundary."""
import numpy as np
import methodlm as M

rng = np.random.default_rng(3)
n = 4000
conf = rng.normal(0, 1, n)
X    = 0.7 * conf + rng.normal(0, 1, n)
med  = 0.9 * X + rng.normal(0, 1, n)
Y    = 0.6 * med + 0.6 * conf + rng.normal(0, 1, n)     # X's effect is fully via med; conf confounds
coll = 1.3 * X + 1.3 * Y + rng.normal(0, 1, n)
noise = rng.normal(0, 1, n)
data = {"X": X, "conf": conf, "med": med, "coll": coll, "noise": noise, "Y": Y}
_, _, _, adjust = M.make_tools(data, "Y", interventional=False)

out = {z: adjust("X", [z]) for z in ["conf", "med", "coll", "noise"]}
for z, s in out.items():
    audit = s.split("[BIAS-AUDIT]")[1].strip() if "[BIAS-AUDIT]" in s else "(none)"
    print(f"X | {z:>5}:  {audit[:120]}")

print()
ok = 0
def check(name, cond):
    global ok
    print(f"  {'OK ' if cond else 'XX '} {name}"); ok += cond

check("collider flagged as COLLIDER", "COLLIDER" in out["coll"])
check("neutral NOT flagged", "no collider signature" in out["noise"])
check("mediator surfaced (explain-away, not collider)", "soak" in out["med"] and "COLLIDER" not in out["med"])
check("confounder surfaced (explain-away, not collider)", "soak" in out["conf"] and "COLLIDER" not in out["conf"])
print(f"\n{ok}/4 bias-audit checks passed")
assert ok == 4, "bias guard regressed"
print("instrument OK -- collider detected, mediator/confounder deferred to the DAG (honest limit)")
