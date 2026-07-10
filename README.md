# MethodLM

**The method, wrapped around a language model, kept honest by a ledger.** Point it at data;
it **computes** on it (a ternary two-timescale readout) and **reasons** about it with the
gbranaa-hue research method — pre-registering every test and keeping an audit trail — so any
model has to *prove* its causal claims instead of asserting them.

> Part of the ternary line — sibling to **[OBSERVE / 012-trit-search](https://github.com/gbranaa4-hue/012-trit-search)**
> (local, private semantic code search). OBSERVE searches your code privately; MethodLM
> reasons about your data honestly. The optional ternary "second witness" here uses the same
> `tritkit` two-timescale layer.

## What it's for

Language models confuse correlation with causation constantly. MethodLM is a **verifiable
causal-reasoning harness**: it makes any model refuse false causation and back its answers
with real tests, leaving a checkable ledger. On a benchmark of confounded scenarios its
backdoor-adjustment test **cuts false causal claims from 100% (naive correlation) to ~1%**
while keeping 100% detection of the true cause (`benchmark_causal.py`).

The discipline — not the model — is the product. A weak local model and a frontier model are
held to the *same* standard: no `FINAL` verdict without a real, pre-registered test.

### Causal tools the copilot can run

| Tool | What it does |
|------|--------------|
| `CORR` | observational correlation (a clue, never a verdict) |
| `RUN` | a true controlled experiment (interventional demo world) |
| `STRAT` | stratified check — does the link survive inside bands of a confounder? |
| `ADJUST` | **backdoor adjustment + sensitivity + bias audit** — effect of X on the target controlling for named confounders, with a Cinelli–Hazlett **robustness value** (how strong a *hidden* confounder would need to be to overturn it; `RV < 0.10` = fragile), **plus a collider/mediator audit** that flags when conditioning on a variable would *introduce* bias (the "Table 2 fallacy"). It detects the data-visible danger (a collider) and honestly defers mediator-vs-confounder to your DAG — so it never tells you to blindly "adjust for everything." |
| `ATTR` | the ternary compute gate's independent evidence per column (the optional second witness) |

Pre-registration is enforced: no `FINAL` is accepted until at least one real test
(`ADJUST`/`STRAT`/`RUN`) has run.

## Install

```
pip install numpy                       # required — the reasoning harness + tools
pip install anthropic                   # optional — to drive a frontier model (--model opus/sonnet/haiku)
pip install pandas pyarrow openpyxl     # optional — extra data formats (Parquet / Excel)
pip install torch                       # optional — enables the ternary second witness
```

- **Reasoning half** runs on `numpy` alone.
- **Frontier backend** needs `anthropic` + `ANTHROPIC_API_KEY` (set a low workspace spend limit).
- **Local backend** needs a llama.cpp `llama-completion` binary + a small GGUF (e.g. Qwen); point `methodlm_models.py` at yours. (Weights/binaries are not shipped here.)
- **Ternary second witness** (optional) needs `torch` + `tritkit` (from
  [012-trit-search](https://github.com/gbranaa4-hue/012-trit-search)); set
  `METHODLM_TRITKIT=/path/to/tritkit_parent`. Without it, MethodLM prints a note and runs the
  reasoning half normally.

## Run it

```
python methodlm.py --demo                       # hidden-confound world + answer key
python methodlm.py --diabetes                   # real data: 442 diabetes patients (sklearn)
python methodlm.py --data FILE --target COLUMN  # any tabular dataset
python methodlm.py --demo --race                # head-to-head vs the same model, no method
python methodlm.py --diabetes --model opus       # drive a frontier model instead of local
python methodlm_gui.py                           # desktop GUI (opens in your browser)
```

## Reasoning backend (`--model`)

| `--model` | Backend | Notes |
|-----------|---------|-------|
| `local` (default) | a GGUF via `llama-completion` | private, offline, free (bring your own model) |
| `opus` / `sonnet` / `haiku` | Claude via the Anthropic API | frontier reasoning; needs `anthropic` + key |

The GUI reads the backend from the `METHODLM_MODEL` environment variable.

## What we measured (multi-model matrix)

`benchmark_models.py` runs plain-vs-harness on the same confounded items across models. The
honest finding: the harness's value is **capability-dependent** — a capable model wrapped in
it reads its own tool output and reaches the correct, auditable driver (where the same model
unwrapped hedges or endorses the decoy); a weak model becomes *safe* (stops confidently
endorsing the bystander) but can't always synthesize a verdict. `test_judge.py` locks the
scorer against real verdicts; every run writes a full-verdict audit JSON for re-scoring.

## Target column

The **target** is the one thing you want explained — the outcome whose *cause* you're after.
MethodLM asks *"what drives the target?"* and treats every other column as a candidate. Run
`--data FILE` with no `--target` to list every column with its kind; the GUI shows them as
clickable chips.

## Data formats (`methodlm_io.py`)

CSV/TSV, JSON/JSONL, Parquet, SQLite (`--table`/`--query`), NumPy `.npz`, Excel. The
featurizer coerces mixed columns and **reports every step** into the ledger; free-text /
high-cardinality columns are dropped (stated, with the count). **Honest boundary:** a lone
image, raw audio, or a free-text blob isn't a "what drives Y" question until something
featurizes it into columns with a target.

## Why this is different from asking a chatbot

Every claim is bought with a test that was pre-registered *before* the result came back and
executed by real computation. When the optional ternary gate (reads gradients) and the
copilot (reads experiments) converge, that's two independent witnesses, not one.

## Honest limits

Small-model reasoning can misread its own results (the ledger catches it; a capable driver
avoids it). Ternary readouts trade precision for ~20× compression. `STRAT` is conditioning,
not intervention — it cannot rule out unmeasured confounds, and the copilot is told so.

## License

MIT — see `LICENSE`.
