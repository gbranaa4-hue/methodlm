#!/usr/bin/env python3
"""MethodLM -- the method, in a language model, kept honest by a ledger.

Point it at data; it computes on it (ternary two-timescale readout) and reasons about
it (gbranaa-hue method), pre-registering every test.

  python methodlm.py --demo                      interventional demo world (hidden
                                                 confound + answer key at the end)
  python methodlm.py --diabetes                  real clinical data (442 patients,
                                                 sklearn load_diabetes, raw units)
  python methodlm.py --csv F --target COL        any recorded CSV (observational)

Pipeline (identical in all modes):
  COMPUTE  a tritkit TwoTimescaleLinear ternary readout learns target from the other
           columns -> NMSE + structural evidence share + gate selection.
  REASON   the method copilot (Qwen-3B + gbranaa-hue method) investigates with tools,
           pre-registers every test, writes an honest ledger.

Tools by mode:  CORR (all) | ATTR (all) | RUN true intervention (demo world only)
                STRAT observational conditioning (recorded data): corr(x,target)
                inside quartile bands of z -- the honest substitute for clamping
                when you cannot rerun the world.
"""
import argparse, os, re, sys, subprocess, textwrap, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_tk = os.environ.get("METHODLM_TRITKIT")     # optional: path to the tritkit pkg (ternary 2nd witness)
if _tk:
    sys.path.insert(0, _tk)
import methodlm_models
rng = np.random.default_rng(11)

BACKEND = None      # the model driving the reasoning; set by main() or lazily to local
def backend():
    global BACKEND
    if BACKEND is None:
        BACKEND = methodlm_models.get_model(os.environ.get("METHODLM_MODEL", "local"), HERE)
    return BACKEND

# ---------------- data sources ----------------
def demo_world(n=2000, clamp=None):
    clamp = clamp or {}
    season = np.cumsum(rng.normal(0, 0.15, n)); season -= season.mean()
    d = {"temperature": 25 + 6 * np.tanh(season) + rng.normal(0, 0.8, n),
         "humidity":    50 + 15 * np.tanh(season) + rng.normal(0, 2.5, n),
         "vibration":   rng.normal(0, 1, n)}
    for k, v in clamp.items():
        if k in d: d[k] = np.full(n, float(v))
    d["error"] = np.clip(0.5 + 0.08 * (d["humidity"] - 50) + rng.normal(0, 0.35, n), 0, None)
    return d

def load_diabetes():
    from sklearn.datasets import load_diabetes as ld
    raw = ld(scaled=False)
    names = ["age", "sex", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"]
    d = {n: raw.data[:, i].astype(float) for i, n in enumerate(names)}
    d["progression"] = raw.target.astype(float)
    return d

def load_csv(path, target):
    import csv as _csv
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(_csv.DictReader(f))
    cols = {}
    for k in rows[0]:
        try:
            cols[k.strip()] = np.array([float(r[k]) for r in rows])
        except ValueError:
            pass                                   # skip non-numeric columns
    assert target in cols, f"target '{target}' not among numeric columns {list(cols)}"
    return cols

# ---------------- COMPUTE: ternary two-timescale readout ----------------
def train_readout(data, target):
    import torch
    import torch.nn.functional as F
    from tritkit.twotimescale import TwoTimescaleLinear
    torch.manual_seed(0)
    names = [k for k in data if k != target]
    X = np.column_stack([data[k] for k in names])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Yv = data[target]; Yn = (Yv - Yv.mean()) / Yv.std()
    Xt, Yt = torch.tensor(X, dtype=torch.float32), torch.tensor(Yn, dtype=torch.float32)
    n = len(Yn)
    lyr = TwoTimescaleLinear(len(names), 1, density=min(0.5, 3 / len(names) + 0.15),
                             bias=True, evidence_beta=0.97)
    opt = torch.optim.SGD(lyr.parameters(), lr=0.02)
    e2 = []
    for ep in range(10):
        perm = torch.randperm(n)
        for t in range(0, n - 16, 16):
            idx = perm[t:t + 16]
            loss = F.mse_loss(lyr(Xt[idx]), Yt[idx, None])
            opt.zero_grad(); loss.backward(); opt.step()
            if t % 320 == 0: lyr.step_gate()
            if ep == 9: e2.append(loss.item())
    ev = lyr.evidence[0].detach().numpy(); share = ev / ev.sum()
    order = np.argsort(-share)
    return (float(np.mean(e2)),
            {names[i]: float(share[i]) for i in order},
            [names[i] for i in range(len(names)) if lyr.G[0, i] > 0])

# ---------------- tools ----------------
def make_tools(data, target, interventional):
    def corr(a, b):
        if a not in data or b not in data:
            return f"unknown column(s); columns are {list(data)}"
        r = float(np.corrcoef(data[a], data[b])[0, 1])
        return f"corr({a},{b}) = {r:+.2f} over {len(data[a])} samples."

    def run(vary, clamp):
        if not interventional:
            return "RUN unavailable: this is recorded data, not a rerunnable system. Use STRAT."
        d = demo_world(400, clamp=clamp)
        r = float(np.corrcoef(d[vary], d[target])[0, 1]) if vary in d else float("nan")
        cl = ", ".join(f"{k}={v}" for k, v in clamp.items()) or "nothing"
        return (f"Controlled run: 400 fresh trials varying {vary}, clamping {cl}. "
                f"corr({vary},{target}) = {r:+.2f}; mean {target} {d[target].mean():.2f}.")

    def strat(x, z):
        if x not in data or z not in data:
            return f"unknown column(s); columns are {list(data)}"
        raw = float(np.corrcoef(data[x], data[target])[0, 1])
        qs = np.quantile(data[z], [0, .25, .5, .75, 1.0])
        rs = []
        for i in range(4):
            m = (data[z] >= qs[i]) & (data[z] <= qs[i + 1])
            if m.sum() > 20 and np.std(data[x][m]) > 0:
                rs.append(float(np.corrcoef(data[x][m], data[target][m])[0, 1]))
        within = float(np.mean(rs)) if rs else float("nan")
        return (f"Stratified: raw corr({x},{target}) = {raw:+.2f}; inside quartile bands of {z} "
                f"it is {['%+.2f' % r for r in rs]} (mean {within:+.2f}). "
                f"If the within-band mean collapses, {x}'s link runs through {z}.")

    def adjust(x, zs):
        """Backdoor adjustment (multiple regression) + Cinelli-Hazlett sensitivity: how
        strong an UNMEASURED confounder would have to be to overturn the adjusted effect.
        ALSO computes the FULL adjustment (every other column) and warns when the requested
        conditioning set omits candidates -- leaving the true confounder out is exactly what
        makes a bystander falsely 'survive'."""
        if x not in data:
            return f"unknown column '{x}'; columns are {list(data)}"
        others = [c for c in data if c not in (x, target)]
        zs = [z for z in zs if z in data and z not in (x, target)]
        n = len(data[target])

        def zsc(a):
            a = np.asarray(a, float); return (a - a.mean()) / (a.std() + 1e-9)
        y = zsc(data[target])

        def fit(cond):                                              # partial corr, t, RV of x | cond
            X = np.column_stack([zsc(data[c]) for c in [x] + cond] + [np.ones(n)])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta; dof = n - X.shape[1]
            se = np.sqrt(((resid ** 2).sum() / max(dof, 1)) * np.diag(np.linalg.pinv(X.T @ X)))
            t = float(beta[0] / (se[0] + 1e-12))
            partial = t / np.sqrt(t * t + dof) if dof > 0 else float("nan")
            f = abs(t) / np.sqrt(max(dof, 1))
            return partial, t, 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)   # RV (q=1)

        partial, t, rv = fit(zs)
        raw = float(np.corrcoef(data[x], data[target])[0, 1])
        zst = ", ".join(zs) if zs else "nothing"
        msg = (f"ADJUST: effect of {x} on {target} controlling for [{zst}] (n={n}). "
               f"raw corr {raw:+.2f} -> adjusted partial corr {partial:+.2f} (t={t:+.1f}). "
               f"Robustness value RV={rv:.2f}: an unmeasured confounder would need to explain "
               f">={rv*100:.0f}% of the residual variance of BOTH {x} and {target} to null it. "
               f"RV<0.10 = fragile; higher = more robust to hidden confounding.")
        omitted = [c for c in others if c not in zs]
        if omitted:                                                # incomplete set -> show the full adjustment
            fp, ft, frv = fit(others)
            flip = (abs(fp) < 0.10) != (abs(partial) < 0.10)
            msg += (f"\n[COMPLETENESS] you left {omitted} OUT of the conditioning set. Backdoor "
                    f"adjustment is only valid controlling for ALL other candidates. Under FULL "
                    f"adjustment {x}'s partial corr is {fp:+.2f} (RV={frv:.2f})"
                    + (f" -- this FLIPS your result: an omitted column was confounding {x}, so its "
                       f"apparent survival was spurious. Trust the FULL adjustment."
                       if flip else ", consistent with your subset."))
        return msg

    return corr, run, strat, adjust

# ---------------- REASON: the copilot loop ----------------
def build_system(cols, target, interventional):
    names = ", ".join(cols)
    a, b = [c for c in cols if c != target][:2]
    runline = (f"  RUN: vary={a}, clamp={{{b}:50}}    (true controlled experiment; clamp value is a NUMBER)"
               if interventional else
               f"  STRAT: {a},{b}          (quick check: corr({a},{target}) inside bands of {b})")
    return textwrap.dedent(f"""\
        You are a research-methodology copilot running a real instrument by the gbranaa-hue
        method. A correlation is never a cause; suspect the boring explanation (artifact,
        confound, the instrument itself) first; trust only tests that could have failed.

        Data columns (EXACT names): {names}. Target: {target}.
        Tools -- end each reply with exactly one tool line:
          CORR: {a},{target}
        {runline}
          ADJUST: {a} | <ALL other columns>   (backdoor adjustment + sensitivity: effect of
                                     {a} on {target} controlling for the listed confounders,
                                     with a robustness value for hidden confounding)
          ATTR:                    (the learned ternary model's evidence per column)
        HOW TO FIND THE DRIVER: for a candidate X, run 'ADJUST: X | <EVERY other candidate
        column>'. Always condition on ALL the others -- omitting even one can leave the true
        confounder in, making a bystander falsely survive; the tool will WARN and show the
        full adjustment if you leave any out, and you must trust that full-adjustment result.
        If X's adjusted partial corr stays large with a high robustness value, X drives
        {target}; if it COLLAPSES toward 0 (or RV < 0.10), X is a confounded bystander.
        CRITICAL: the driver is the candidate that SURVIVES full adjustment -- it is NEVER
        the variable you controlled for. Do not name a conditioning/control variable as the
        cause; that is backwards. Strategy: ADJUST the tempting/decoy variable
        first; if it collapses, ADJUST the other strong candidate to confirm the real
        driver, then conclude.
        Before any ADJUST/STRAT/RUN, include a 'PREREGISTER:' line in the SAME reply naming
        the same X you test and what result confirms vs disconfirms. Comparing two
        correlations is NOT a test. You MUST run at least one ADJUST/STRAT/RUN (never only
        CORR) before any FINAL. Your FINAL must name the variable whose adjusted effect
        SURVIVED as the driver (or say none did). Be brief.""")

def ask(system, messages, n=230):
    return backend().generate(system, messages, n)

def vanilla_answer(question, n=200):
    """The SAME model, no method and no tools -- a plain analyst. The control racer."""
    sysp = ("You are a helpful data analyst. Answer the question directly and concisely: "
            "give your conclusion and a recommendation.")
    return ask(sysp, [{"role": "user", "content": question}], n)

def investigate(name, data, target, question, interventional, answer_key=None, ingest_report=None):
    ledger = os.path.join(HERE, f"ledger_{name}.txt")
    led = open(ledger, "w", encoding="utf-8")
    def out(s):
        led.write(s + "\n"); led.flush()
        try: print(s, flush=True)
        except UnicodeEncodeError: print(s.encode("ascii", "replace").decode(), flush=True)

    if ingest_report:
        out(f"[ingest]\n{ingest_report}\n")
    t0 = time.time()
    try:
        nmse, share, gate = train_readout(data, target)
        top = ", ".join(f"{k} {v*100:.0f}%" for k, v in list(share.items())[:4])
        out(f"[compute] ternary readout NMSE {nmse:.2f} | evidence: {top} | gate: {gate}")
    except ImportError:
        nmse, share, gate = None, {}, []
        out("[compute] ternary second witness skipped (set METHODLM_TRITKIT + install torch to enable)")
    corr, run, strat, adjust = make_tools(data, target, interventional)
    system = build_system(list(data), target, interventional)

    msgs = [{"role": "user", "content": question}]
    pre = nrun = 0
    verdict = "(no verdict — ran out of turns)"
    seen = {}          # loop guard: signatures of tests already run
    stuck = 0          # consecutive turns producing no usable tool call
    for turn in range(1, 10):
        raw = ask(system, msgs)
        # ENFORCE ONE ACTION PER TURN: keep text up to and including the first tool
        # directive (or FINAL); drop anything the model dumped after it.
        cut = None
        for m in re.finditer(r"^\s*(CORR|RUN|STRAT|ADJUST|ATTR|FINAL)\b.*$", raw, re.MULTILINE | re.IGNORECASE):
            cut = m.end(); break
        reply = raw[:cut] if cut else raw
        out(f"\n--- copilot turn {turn} ---\n{reply}")
        msgs.append({"role": "assistant", "content": reply})
        if "PREREGISTER:" in reply: pre += 1
        m_run = re.search(r"RUN:\s*vary=(\w+),\s*clamp=\{([^}]*)\}", reply)
        m_adj = re.search(r"ADJUST:\s*(\w+)\s*\|\s*([\w,\s]*)", reply)
        m_str = re.search(r"STRAT:\s*(\w+)\s*,\s*(\w+)", reply)
        m_cor = re.search(r"CORR:\s*(\w+)\s*,\s*(\w+)", reply)
        m_attr = re.search(r"\bATTR:", reply)
        # FINAL honored ONLY when it's not a hedge AND a real test has run
        if re.search(r"\bfinal\s*:", reply, re.IGNORECASE) and not (m_run or m_adj or m_str or m_cor or m_attr):
            if nrun < 1:
                out("[TOOL] REFUSED: comparing correlations is not a test. Run one STRAT or "
                    "RUN before concluding.")
                msgs.append({"role": "user", "content": "REFUSED: run at least one STRAT or "
                             "RUN test before FINAL."})
                continue
            verdict = re.sub(r"^.*?final\s*:", "", reply, flags=re.IGNORECASE | re.DOTALL).strip()
            out("\n[done] verdict reached."); break
        if m_run:
            clamp = {k: float(v) for k, v in re.findall(r"(\w+)\s*:\s*([-\d.]+)", m_run.group(2))}
            res = ("REFUSED: PREREGISTER in the same reply first." if "PREREGISTER:" not in reply
                   else "Clamp needs numbers, e.g. clamp={humidity:50}" if not clamp
                   else run(m_run.group(1), clamp)); nrun += m_run and bool(clamp) and "PREREGISTER:" in reply
        elif m_adj:
            zs = [z.strip() for z in m_adj.group(2).split(",") if z.strip()]
            res = ("REFUSED: PREREGISTER in the same reply first." if "PREREGISTER:" not in reply
                   else adjust(m_adj.group(1), zs)); nrun += "PREREGISTER:" in reply
        elif m_str:
            res = ("REFUSED: PREREGISTER in the same reply first." if "PREREGISTER:" not in reply
                   else strat(m_str.group(1), m_str.group(2))); nrun += bool(m_str) and "PREREGISTER:" in reply
        elif m_cor:
            res = corr(m_cor.group(1), m_cor.group(2))
        elif re.search(r"\bATTR:", reply):
            if nmse is None:
                res = "Ternary second witness unavailable (install torch + set METHODLM_TRITKIT)."
            else:
                res = (f"Ternary readout (NMSE {nmse:.2f}) evidence share: "
                       + ", ".join(f"{k}: {v*100:.0f}%" for k, v in share.items())
                       + f". Gate connects: {', '.join(gate)}.")
        else:
            res = "No tool recognized. Use CORR:, STRAT:/RUN:, ATTR:, or FINAL:."
        # LOOP GUARD: a weak model can re-run the same test forever. If a real test
        # repeats, don't re-run it -- nudge to conclude; force a stop on a 2nd repeat.
        sig = next((g.groups() for g in (m_run, m_adj, m_str, m_cor) if g), None)
        if sig and not str(res).startswith(("REFUSED", "Clamp")):
            if sig in seen:
                seen[sig] += 1
                # nudge toward the NEXT untested candidate (not FINAL) -- a collapsed test
                # means that variable is a bystander, not that the job is done.
                tested = ", ".join(sorted({s[0] for s in seen})) or "none"
                untested = [c for c in data if c not in (target,) and c not in {s[0] for s in seen}]
                res = (f"{res}\n[REPEAT: you already ran this. A collapsed effect (RV<0.10) means "
                       f"that variable is a BYSTANDER, not the answer. You have tested: {tested}. "
                       f"Now run ADJUST on a DIFFERENT untested candidate ({', '.join(untested) or 'none left'}) "
                       "to find the real driver. Reply FINAL only once a variable SURVIVES (high RV).]")
                if seen[sig] >= 3:
                    out(f"[TOOL] {res}")
                    verdict = "(loop-guard: model repeated the same test without concluding)"
                    out("\n[done] loop-guard stopped a repeat loop."); break
            else:
                seen[sig] = 1
        out(f"[TOOL] {res}")
        # NO-PROGRESS guard: a model that can't emit tool syntax loops uselessly.
        if str(res).startswith(("No tool recognized", "REFUSED")):
            stuck += 1
            if stuck >= 3:
                verdict = "(model could not drive the tool protocol)"
                out("\n[done] no-progress guard: model never produced a usable test."); break
        else:
            stuck = 0
        msgs.append({"role": "user", "content": res})
    if answer_key:
        out(f"\n[ANSWER KEY -- the instrument was never told] {answer_key}")
    out(f"\n[ledger] preregistrations {pre}, registered tests {nrun}, "
        f"{time.time()-t0:.0f}s -> {ledger}")
    led.close()
    return {"verdict": verdict, "pre": pre, "nrun": nrun,
            "nmse": nmse, "gate": gate, "share": share}


def finish_race(question, res, answer_key=None):
    """Same question, two racers: MethodLM (method + tools + tests) vs plain LLM."""
    print("\n" + "=" * 62)
    print("HEAD-TO-HEAD -- same question, two racers")
    print("=" * 62)
    v = vanilla_answer(question)
    tested = f"{res['nrun']} test(s), {res['pre']} pre-reg" if res["nrun"] else "NO TEST RUN"
    print(f"\n[ vanilla 3B | no method, no tools ]\n  {v}\n")
    print(f"[ MethodLM | method + tools | {tested} ]\n  {res['verdict']}\n")
    if answer_key:
        print(f"[ answer key ] {answer_key}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--diabetes", action="store_true")
    ap.add_argument("--data", help="any file: csv/tsv/json/jsonl/parquet/sqlite/npz/xlsx")
    ap.add_argument("--csv", help="alias for --data")
    ap.add_argument("--folder", help="labelled folder of text/image files")
    ap.add_argument("--target")
    ap.add_argument("--table", help="sqlite table name (optional)")
    ap.add_argument("--query", help="sqlite SQL query (optional)")
    ap.add_argument("--race", action="store_true", help="also run a plain-LLM racer on the same question")
    ap.add_argument("--model", default="local", help="reasoning backend: local (Qwen-3B) | claude")
    args = ap.parse_args()
    global BACKEND
    BACKEND = methodlm_models.get_model(args.model, HERE)
    print(f"[methodlm] reasoning backend: {BACKEND.label}")
    if args.demo:
        data = demo_world()
        r = float(np.corrcoef(data['temperature'], data['error'])[0, 1])
        q = (f"Our sensor's error correlates with temperature (r={r:+.2f}) in 2000 logged "
             "samples; the team wants to install cooling. Find what actually drives the "
             "error before we spend the money.")
        key = "error = f(humidity); temperature only co-rises with humidity via season."
        res = investigate("demo", data, "error", q, True, key)
        if args.race: finish_race(q, res, key)
    elif args.diabetes:
        data = load_diabetes()
        r = float(np.corrcoef(data['bmi'], data['progression'])[0, 1])
        q = (f"In 442 real diabetes patients, bmi correlates with one-year disease "
             f"progression (r={r:+.2f}). The clinic wants to fund a weight-loss-only "
             "program. Before they do: is bmi's link robust, or does it run through blood "
             "serum markers like ltg? You cannot rerun patients; use STRAT.")
        res = investigate("diabetes", data, "progression", q, False)
        if args.race: finish_race(q, res)
    elif args.folder and args.target:
        from methodlm_io import load_folder, featurize, format_report
        raw, notes = load_folder(args.folder)
        data, rep = featurize(raw, args.target)
        report = format_report(notes, rep, args.target)
        cols = [c for c in data if c != args.target]
        q = (f"Investigate what actually drives {args.target} across these files "
             f"(features: {', '.join(cols[:12])}). Do not trust raw correlations.")
        res = investigate(os.path.basename(args.folder.rstrip('/\\')) or "folder",
                          data, args.target, q, False, ingest_report=report)
        if args.race: finish_race(q, res)
    elif args.data or args.csv:
        from methodlm_io import validate, load_any, featurize, format_report
        path = args.data or args.csv
        v = validate(path, args.target, table=args.table, query=args.query)
        if not v["ok"]:
            print("MethodLM cannot run yet:")
            for e in v["errors"]:
                print(f"  x {e}")
            if v["suggestions"]:
                print(f"  -> try: {', '.join(map(str, v['suggestions']))}")
            if v["info"]:
                print("  columns available: "
                      + ", ".join(f"{c['name']} [{c['kind']}]" for c in v["info"]["columns"]))
            return
        raw, notes = load_any(path, table=args.table, query=args.query)
        data, rep = featurize(raw, args.target)
        report = format_report(notes, rep, args.target)
        cols = [c for c in data if c != args.target]
        q = (f"Investigate what actually drives {args.target} in this recorded dataset "
             f"(columns: {', '.join(cols[:12])}{'...' if len(cols) > 12 else ''}). "
             "Do not trust raw correlations.")
        name = os.path.splitext(os.path.basename(path))[0]
        res = investigate(name, data, args.target, q, False, ingest_report=report)
        if args.race: finish_race(q, res)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
