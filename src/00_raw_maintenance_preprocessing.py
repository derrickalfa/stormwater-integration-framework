"""
00_raw_maintenance_preprocessing.py
=====================================
Filters raw maintenance work order records from xlsb format.
Splits multi-activity descriptions into individual sentences and
retains only those containing recognised maintenance keywords.

Input:  Asset maintenance record.xlsb (raw DSD work orders)
Output: CLeanedAllSheets.xlsx (filtered, one row per activity)

Author: Research Team
"""
import re, os, sys, pandas as pd
from pathlib import Path
from pyxlsb import open_workbook
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Paths, Params

SHEETS = ["Export Worksheet", "Sheet1", "Sheet2", "Sheet3", "Sheet4", "Sheet5"]

INCLUDE_KEYWORDS = [
    "clear", "clean", "cleaning", "cleansing", "clearing", "repair", "lining", "cured",
    "sleeve", "cutting", "choked", "demolish", "construct", "construction", "desilt",
    "precast", "concrete", "cipp", "c.i.p.p", "cut", "hole", "holes", "excavate",
    "excavation", "cement", "desludging", "structural improvement", "seal", "sealing",
    "block", "blockage", "trench", "modify", "transplant", "tree", "vegetation",
    "lay", "laying", "reinstate", "abandon", "infiltration", "leak", "rust", "frame",
]
WORD_PAIRS = [("cctv", "carry"), ("cctv", "conduct")]

INCLUDE_PAT = re.compile(r"\b(" + "|".join(re.escape(k) for k in INCLUDE_KEYWORDS) + r")\b", re.I)
PAIR_PATS   = [(re.compile(rf"\b{re.escape(w1)}\b", re.I), re.compile(rf"\b{re.escape(w2)}\b", re.I))
               for w1, w2 in WORD_PAIRS]

PROTECT = [
    (re.compile(r"\b(\d+\.\d+)\s*(m2|mm|m|nr|nrs|nos?)\.?\b"), r"__DEC_\1_\2__"),
    (re.compile(r"\b(\d+)\s*(m2|mm|m|nr|nrs|nos?)\.?\b"),       r"__WHO_\1_\2__"),
    (re.compile(r"\b(n\.e\.)"), "__NE__"), (re.compile(r"\b(dia\.)"), "__DIA__"),
]
RESTORE = [
    (re.compile(r"__DEC_(\d+\.\d+)_(m2|mm|m|nr|nrs|nos?)__"), r"\1 \2"),
    (re.compile(r"__WHO_(\d+)_(m2|mm|m|nr|nrs|nos?)__"),       r"\1 \2"),
]
SPLIT_PAT = re.compile(r"""
    (?=\b\d+[\.\)\]\/,;]) | (?=Remarks?:) | (?=[-•*]+\s) |
    (?<!\bto\s)(?<!\band\s)(?=\bProvide\b|\bSupply\b|\bCarry\b|\bClear\b|\bConduct\b) |
    ; | (?<=\.)\s+(?=[A-Z]) | \n\s*\n
""", re.VERBOSE | re.I)


def _split(text):
    if not isinstance(text, str) or not text.strip(): return []
    t = text
    for pat, rep in PROTECT: t = pat.sub(rep, t)
    parts = [p.strip() for p in SPLIT_PAT.split(t) if p.strip() and len(p.strip()) >= Params.MIN_ACTIVITY_LENGTH]
    out = []
    for p in parts:
        for pat, rep in RESTORE: p = pat.sub(rep, p)
        p = p.replace("__NE__", "n.e.").replace("__DIA__", "dia.")
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) >= Params.MIN_ACTIVITY_LENGTH: out.append(p)
    return out


@lru_cache(maxsize=10_000)
def _has_kw(text): return INCLUDE_PAT.search(text) is not None


def _filter(sentences):
    any_kw = any(_has_kw(s.lower()) for s in sentences)
    kept = []
    for s in sentences:
        sl = s.lower()
        if "ditto" in sl:
            if any_kw: kept.append(s)
        elif _has_kw(sl) or any(p1.search(sl) and p2.search(sl) for p1, p2 in PAIR_PATS):
            kept.append(s)
    return kept


def _process_batch(args):
    batch, target_col = args
    results = []
    for idx, row in batch:
        kept = _filter(_split(row.get(target_col, "")))
        results.append((idx, "\n".join(kept)))
    return results


def main():
    Paths.ensure_output_dirs()
    target_col = "Description of Works"
    n_workers  = max(1, os.cpu_count() - 1)

    with open_workbook(str(Paths.RAW_MAINTENANCE)) as wb:
        available = wb.sheets

    with pd.ExcelWriter(str(Paths.CLEANED_MAINTENANCE), engine="openpyxl") as writer:
        for sheet in SHEETS:
            if sheet not in available:
                print(f"  ⚠  Sheet '{sheet}' not found — skipped"); continue
            print(f"\nProcessing: {sheet}")
            df = pd.read_excel(str(Paths.RAW_MAINTENANCE), engine="pyxlsb", sheet_name=sheet)
            combined = [""] * len(df)

            rows    = list(df.iterrows())
            batches = [rows[i:i+100] for i in range(0, len(rows), 100)]
            with ProcessPoolExecutor(n_workers) as ex:
                futs = {ex.submit(_process_batch, (b, target_col)): b for b in batches}
                for fut in as_completed(futs):
                    for idx, text in fut.result():
                        combined[idx] = text

            df[target_col] = combined
            df = df[df[target_col].str.strip() != ""]
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
            print(f"  ✓ {len(df):,} rows saved")

    print(f"\nSaved → {Paths.CLEANED_MAINTENANCE}")


if __name__ == "__main__":
    main()
