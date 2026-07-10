"""MethodLM ingestion -- convert many data formats into the canonical table
(numeric columns + a target) that both halves of MethodLM need, and REPORT every
transformation so the ledger shows exactly what was done to the data.

load_any(path)         -> (raw column dict, load notes)   dispatch on extension
featurize(cols,target) -> (numeric column dict, report)   coerce mixed -> numeric

Supported now: .csv .tsv .txt, .json, .jsonl/.ndjson, .parquet/.pq, .db/.sqlite,
.npz, .xlsx (needs openpyxl). Featurizer handles: numeric, boolean, datetime ->
date parts, numeric-looking strings, binary + low-card categoricals (one-hot),
and DROPS free text / high-cardinality columns (stated, with the count).

BOUNDARY (honest): MethodLM asks "what drives the target?" -- it needs columns and a
target. A lone image, raw audio, or a free-text document is not that until something
featurizes it into columns with a target. This layer does the tabular conversions;
it does not pretend a single unlabeled blob is a causal question.
"""
import os, re, sqlite3, warnings
import numpy as np

SUPPORTED_EXT = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson", ".parquet",
                 ".pq", ".db", ".sqlite", ".sqlite3", ".npz", ".xlsx", ".xls"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
TXT_EXT = {".txt", ".md", ".log"}

_DATEISH = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}:\d{2}")

def _looks_datelike(series):
    sample = series.dropna().astype(str).head(20)
    return len(sample) > 0 and (sample.str.contains(_DATEISH).mean() > 0.7)


def load_any(path, table=None, query=None):
    ext = os.path.splitext(path)[1].lower()
    notes = []
    import pandas as pd
    if ext in (".csv", ".txt"):
        df = pd.read_csv(path, sep=None, engine="python")
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif ext == ".json":
        df = pd.read_json(path)
    elif ext in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True)
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(path)
        except ImportError:
            raise SystemExit("Excel support needs:  pip install openpyxl")
    elif ext in (".db", ".sqlite", ".sqlite3"):
        con = sqlite3.connect(path)
        if query:
            df = pd.read_sql_query(query, con)
        else:
            if not table:
                tabs = pd.read_sql_query(
                    "SELECT name FROM sqlite_master WHERE type='table'", con)["name"].tolist()
                if not tabs:
                    raise SystemExit("no tables in sqlite file")
                table = tabs[0]
                notes.append(f"sqlite: auto-selected table '{table}' (of {tabs})")
            df = pd.read_sql_query(f"SELECT * FROM {table}", con)
        con.close()
    elif ext == ".npz":
        z = np.load(path)
        df = pd.DataFrame({k: z[k].ravel() for k in z.files})
        notes.append(f"npz arrays -> columns {list(z.files)}")
    else:
        raise SystemExit(f"unsupported extension '{ext}'. Convert to CSV/JSON/Parquet/SQLite.")
    return {c: df[c].to_numpy() for c in df.columns}, notes


def featurize(cols, target):
    import pandas as pd
    num, rep = {}, {"kept": [], "encoded": [], "derived": [], "dropped": []}

    def add(name, series):
        v = pd.to_numeric(series, errors="coerce")
        num[name] = v.fillna(v.mean()).astype(float).to_numpy()

    for name, arr in cols.items():
        if name == target:
            continue
        s = pd.Series(arr)
        if pd.api.types.is_bool_dtype(s):
            num[name] = s.astype(float).to_numpy(); rep["encoded"].append(f"{name}: bool->0/1"); continue
        if pd.api.types.is_numeric_dtype(s):
            add(name, s); rep["kept"].append(name); continue
        co = pd.to_numeric(s, errors="coerce")
        if co.notna().mean() > 0.9:
            add(name, s); rep["kept"].append(name); continue
        if _looks_datelike(s):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dt = pd.to_datetime(s, errors="coerce")
            if dt.notna().mean() > 0.8:
                made = []
                for part, vals in [("year", dt.dt.year), ("month", dt.dt.month),
                                   ("dow", dt.dt.dayofweek), ("hour", dt.dt.hour)]:
                    if vals.nunique(dropna=True) > 1:
                        num[f"{name}_{part}"] = vals.fillna(vals.median()).astype(float).to_numpy()
                        made.append(part)
                rep["derived"].append(f"{name}: datetime -> {made}"); continue
        nun = s.nunique(dropna=True)
        if nun == 2:
            cats = sorted(map(str, s.dropna().unique()))
            num[name] = (s.astype(str) == cats[-1]).astype(float).to_numpy()
            rep["encoded"].append(f"{name}: binary {cats}"); continue
        if 2 < nun <= 10:
            for c in sorted(map(str, s.dropna().unique())):
                num[f"{name}={c}"] = (s.astype(str) == c).astype(float).to_numpy()
            rep["encoded"].append(f"{name}: one-hot ({nun} levels)"); continue
        rep["dropped"].append(f"{name}: {nun} categories / free text")

    # target
    ts = pd.Series(cols[target])
    ct = pd.to_numeric(ts, errors="coerce")
    if ct.notna().mean() > 0.9:
        num[target] = ct.fillna(ct.mean()).astype(float).to_numpy()
    elif ts.nunique(dropna=True) == 2:
        cats = sorted(map(str, ts.dropna().unique()))
        num[target] = (ts.astype(str) == cats[-1]).astype(float).to_numpy()
        rep["encoded"].append(f"TARGET {target}: binary {cats} -> {cats[-1]}=1")
    else:
        raise SystemExit(f"target '{target}' is neither numeric nor binary -- "
                         f"MethodLM needs a target it can pose 'what drives it' about "
                         f"({ts.nunique()} distinct values found).")

    n = min(len(v) for v in num.values())
    return {k: v[:n] for k, v in num.items()}, rep


def describe(path, table=None, query=None):
    """Cheap column census for validation / GUI dropdowns -- names, inferred kinds,
    row count, and which columns can serve as a target."""
    import pandas as pd
    raw, notes = load_any(path, table=table, query=query)
    cols, n = [], 0
    for name, arr in raw.items():
        s = pd.Series(arr); n = max(n, len(s))
        if pd.api.types.is_bool_dtype(s):
            kind = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "numeric"
        elif pd.to_numeric(s, errors="coerce").notna().mean() > 0.9:
            kind = "numeric-text"
        elif _looks_datelike(s):
            kind = "datetime"
        else:
            nun = int(s.nunique(dropna=True))
            kind = "binary" if nun == 2 else ("categorical" if nun <= 10 else f"text({nun})")
        cols.append({"name": name, "kind": kind})
    cands = [c["name"] for c in cols if c["kind"] in ("numeric", "numeric-text", "boolean", "binary")]
    return {"columns": cols, "n_rows": n, "notes": notes, "target_candidates": cands}


def validate(path, target=None, table=None, query=None):
    """Return {ok, errors, suggestions, info} -- so a caller can reject bad input and
    show the user exactly what IS valid instead of a stack trace."""
    import os, difflib
    out = {"ok": False, "errors": [], "suggestions": [], "info": None}
    if not os.path.exists(path):
        out["errors"].append(f"file not found: {path}")
        return out
    ext = os.path.splitext(path)[1].lower()
    if os.path.isfile(path) and ext not in SUPPORTED_EXT:
        out["errors"].append(f"unsupported format '{ext}'")
        out["suggestions"] = ["convert to .csv, .json, .parquet, or .sqlite",
                              "or use --folder for a labelled folder of files"]
        return out
    try:
        info = describe(path, table, query)
    except SystemExit as e:
        out["errors"].append(str(e)); return out
    except Exception as e:
        out["errors"].append(f"could not read file: {e}"); return out
    out["info"] = info
    cols = [c["name"] for c in info["columns"]]
    if not target:
        out["errors"].append("choose a target column")
        out["suggestions"] = info["target_candidates"][:8]
        return out
    if target not in cols:
        near = difflib.get_close_matches(target, cols, n=5, cutoff=0.3)
        out["errors"].append(f"target '{target}' is not a column")
        out["suggestions"] = near or info["target_candidates"][:8]
        return out
    if target not in info["target_candidates"]:
        out["errors"].append(f"target '{target}' is text/high-cardinality; MethodLM needs "
                             f"a numeric or binary target")
        out["suggestions"] = info["target_candidates"][:8]
        return out
    out["ok"] = True
    return out


def load_folder(root, label="__dir__"):
    """A labelled folder of files -> a feature table. Text files become lexical features,
    images become basic pixel-statistic features; the label is the parent folder name
    (so root/spam/*.txt vs root/ham/*.txt gives label=spam/ham). The honest bridge from
    non-tabular files to MethodLM's 'columns + target' question."""
    import glob
    import numpy as np
    rows = []
    for path in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        lab = os.path.basename(os.path.dirname(path)) if label == "__dir__" else label
        if ext in TXT_EXT:
            t = open(path, encoding="utf-8", errors="ignore").read()
            w = t.split()
            rows.append({"n_chars": len(t), "n_words": len(w),
                         "avg_word_len": float(np.mean([len(x) for x in w])) if w else 0.0,
                         "digit_share": sum(c.isdigit() for c in t) / max(len(t), 1),
                         "upper_share": sum(c.isupper() for c in t) / max(len(t), 1),
                         "label": lab})
        elif ext in IMG_EXT:
            from PIL import Image
            im = Image.open(path).convert("RGB")
            a = np.asarray(im, dtype=float) / 255.0
            rows.append({"width": im.width, "height": im.height,
                         "aspect": im.width / max(im.height, 1),
                         "r_mean": float(a[..., 0].mean()), "g_mean": float(a[..., 1].mean()),
                         "b_mean": float(a[..., 2].mean()), "brightness": float(a.mean()),
                         "contrast": float(a.std()), "label": lab})
    if not rows:
        raise SystemExit(f"no .txt/.md or image files found under {root}")
    keys = rows[0].keys()
    return {k: np.array([r.get(k, np.nan) for r in rows], dtype=object if k == "label" else float)
            for k in keys}, [f"folder {root}: {len(rows)} files -> {len(keys)-1} features + label"]


def format_report(notes, rep, target):
    lines = []
    for nt in notes:
        lines.append(f"  {nt}")
    if rep["kept"]:
        lines.append(f"  kept numeric ({len(rep['kept'])}): {', '.join(rep['kept'])}")
    for k in ("encoded", "derived", "dropped"):
        for item in rep[k]:
            lines.append(f"  {k}: {item}")
    lines.append(f"  target: {target}")
    return "\n".join(lines)
