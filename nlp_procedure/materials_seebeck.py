# app.py
import streamlit as st
import re
import sqlite3
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
from math import isnan

# Try better PDF parser first
try:
    import pdfplumber
    PDF_PLUMBER_AVAILABLE = True
except Exception:
    PDF_PLUMBER_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except Exception:
    PYPDF2_AVAILABLE = False

# Try pymatgen for formula parsing (user asked for pymagen -> use pymatgen)
try:
    from pymatgen.core.composition import Composition
    PYMATGEN_AVAILABLE = True
except Exception:
    PYMATGEN_AVAILABLE = False

st.title("PDF → SQLite / CSV: Greedy NER for Seebeck Coefficient (uses pymatgen for formulas)")

st.markdown(
    """
Upload a PDF. The app will:
- find every occurrence of the phrase **'Seebeck coefficient'**,
- perform a greedy numeric + unit search around every occurrence,
- attempt to extract the corresponding material name or chemical formula found just before the value,
- validate chemical formulas with `pymatgen.Composition` when available,
- save results to a SQLite `.db` and CSV.
"""
)

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
window_chars = st.slider("Context window (characters around 'Seebeck coefficient')", 80, 2000, 600, 10)
max_material_tokens = st.slider("Max tokens to consider for material name before value", 1, 10, 5)
outlier_zscore_thresh = st.number_input("Z-score threshold to flag outliers", 0.0, 10.0, 2.5, 0.1)

if uploaded_file is None:
    st.info("Upload a PDF to start.")
    st.stop()

# --- extract text robustly ---
def extract_text_from_pdf(filelike):
    text = ""
    filelike.seek(0)
    if PDF_PLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(filelike) as pdf:
                for p in pdf.pages:
                    page_text = p.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception:
            pass
    # fallback to PyPDF2
    if PYPDF2_AVAILABLE:
        try:
            filelike.seek(0)
            reader = PdfReader(filelike)
            for page in reader.pages:
                try:
                    t = page.extract_text()
                except Exception:
                    t = ""
                if t:
                    text += t + "\n"
            return text
        except Exception:
            pass
    # Last resort: read bytes and decode naive
    try:
        filelike.seek(0)
        text = filelike.read().decode('utf-8', errors='ignore')
    except Exception:
        text = ""
    return text

text = extract_text_from_pdf(uploaded_file)

if len(text.strip()) == 0:
    st.error("Failed to extract text from PDF (no text found). Try a different PDF or install pdfplumber for better extraction.")
    st.stop()

st.subheader("Preview (first 2000 chars)")
st.text_area("extracted_text_preview", text[:2000], height=220)

# --- patterns ---
# Numeric pattern: allow ±, scientific, commas, ranges like "120–140 µV/K" or "120 μV K^-1"
num_pattern = r"(?P<number>[+\-]?\d{1,3}(?:[,\d{3}])*\.?\d*(?:[eE][+\-]?\d+)?|\d*\.?\d+(?:[eE][+\-]?\d+)?)"
# units pattern: common
units_pattern = r"(?:µV\/K|μV\/K|uV\/K|mV\/K|V\/K|μV K(?:\^-1|⁻¹)?|uV K(?:\^-1|⁻¹)?|mV K(?:\^-1|⁻¹)?)"
# full value pattern including optional ± and ranges (greedy)
value_regex = re.compile(
    rf"(?:(?:±|\+\/\-)\s*)?(?:{num_pattern})(?:\s*(?:to|–|-|—|–)\s*{num_pattern})?\s*(?P<unit>{units_pattern})",
    flags=re.IGNORECASE
)

# Find all occurrences of 'seebeck coefficient'
sc_occurrences = [m for m in re.finditer(r"seebeck coefficient", text, flags=re.IGNORECASE)]
st.write(f"Found **{len(sc_occurrences)}** occurrences of the phrase 'seebeck coefficient' (case-insensitive).")

# Helper: find nearest numeric match to an index within a search window
def find_nearest_value(center_idx, txt, window=600):
    start = max(0, center_idx - window)
    end = min(len(txt), center_idx + window)
    snippet = txt[start:end]
    matches = []
    for m in value_regex.finditer(snippet):
        # absolute position in full text
        abs_start = start + m.start()
        abs_end = start + m.end()
        dist = min(abs_start - center_idx if abs_start >= center_idx else center_idx - abs_end,
                   abs_end - center_idx if abs_end >= center_idx else center_idx - abs_start)
        # store numeric groups; handle ranges by taking the first number as representative and keep range
        gdict = m.groupdict()
        raw = snippet[m.start():m.end()]
        matches.append({
            "abs_start": abs_start,
            "abs_end": abs_end,
            "dist": abs(abs(center_idx - (abs_start + abs_end)//2)),
            "raw": raw.strip(),
            "match_obj": m,
            "snippet": snippet,
        })
    # sort by distance
    matches_sorted = sorted(matches, key=lambda x: x["dist"])
    return matches_sorted

# Helper: attempt to find material/formula preceding a value index
# Strategy:
#  - look at up to N tokens (space-separated) before the numeric match
#  - accept candidates that look like chemical formulas (e.g., Bi2Te3) validated by pymatgen
#  - otherwise choose previous capitalized multi-word phrase (e.g., "Bi2Te3-based material", "Copper (Cu)")
def find_material_before(idx_start, txt, max_tokens=5):
    # get text up to idx_start
    upto = txt[max(0, idx_start - 400): idx_start + 50]  # a bit after number as fallback
    # tokenization by common separators
    tokens = re.split(r"(\s+|,|;|\(|\)|:|\[|\]|\-|—|–)", upto)
    # build candidate tokens (filter out whitespace/separators)
    words = [t for t in tokens if t and not re.match(r"^\s+$", t)]
    # reverse iterate to build candidate phrases with up to max_tokens words
    candidates = []
    # attempt to form tokens by ignoring pure separators
    cleaned_words = [w.strip() for w in words if w.strip() != ""]
    # start from last cleaned word before number
    for n in range(1, max_tokens + 1):
        if len(cleaned_words) == 0:
            break
        # last n words
        cand = " ".join(cleaned_words[-n:])
        candidates.append(cand)
    # Also consider the immediately previous sentence (split by .!?)
    prev_sentences = re.split(r"[\.!?]\s+", upto)
    if len(prev_sentences) > 0:
        prev_sentence = prev_sentences[-1].strip()
        if prev_sentence:
            candidates.append(prev_sentence)
    # Validate candidates: prefer formulas validated by pymatgen
    validated = []
    for c in candidates:
        c_clean = c.strip()
        # Try extract probable formula with a formula regex: element caps followed by numbers e.g. Bi2Te3, Cu, Fe2O3
        formula_like = re.search(r"\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+|[A-Z][a-z]?\d*)\b", c_clean)
        if formula_like:
            cand_formula = formula_like.group(0)
            if PYMATGEN_AVAILABLE:
                try:
                    comp = Composition(cand_formula)
                    # if parsing gives a non-empty composition, accept
                    if sum(comp.get_el_amt_dict().values()) > 0:
                        return cand_formula, "formula_validated"
                except Exception:
                    pass
            else:
                # Without pymatgen, still accept formula-like strings (heuristic)
                return cand_formula, "formula_heuristic"
    # If no formula found, pick the most recent noun phrase candidate: look for capitalised words / multiword names
    for c in candidates:
        # heuristics: contains capitalized word or 'all-lower' words but hyphenated
        if re.search(r"[A-Z][a-z]{2,}", c):
            sm = c.strip()
            # sanitize
            sm = re.sub(r"[^A-Za-z0-9\-\s\(\)\[\],\.]", "", sm)
            return sm, "name_heuristic"
    # fallback: unknown
    return "", "none_found"

# --- process each occurrence ---
rows = []
for occ in sc_occurrences:
    center_idx = occ.start()
    # greedy search for values in the window
    candidates = find_nearest_value(center_idx, text, window=window_chars)
    if not candidates:
        # try expanding window if none
        candidates = find_nearest_value(center_idx, text, window=window_chars*2)
    if not candidates:
        rows.append({
            "seebeck_phrase_idx": center_idx,
            "seebeck_context": text[max(0, center_idx-80): center_idx+80],
            "value_raw": None,
            "value_numeric": None,
            "unit": None,
            "material": None,
            "material_source": "no_value_found",
        })
        continue
    # take top candidate (closest) but keep raw and also allow parsing
    top = candidates[0]
    raw = top["raw"]
    # parse numeric and unit from raw using value_regex again
    m = value_regex.search(raw)
    if m:
        unit = m.group("unit")
        # extract first number (handle ranges — take first, but keep raw for info)
        # group "number" corresponds to the first captured numeric group
        number_text = m.group(1)
        # normalized number: remove commas
        number_text_norm = number_text.replace(",", "")
        try:
            numeric_val = float(number_text_norm)
        except Exception:
            # try to extract digits manually
            try:
                numeric_val = float(re.sub(r"[^\d\.\-eE+]", "", number_text_norm))
            except Exception:
                numeric_val = None
    else:
        unit = None
        numeric_val = None

    # find associated material (search before the numeric match absolute start)
    mat, mat_src = find_material_before(top["abs_start"], text, max_tokens=max_material_tokens)

    rows.append({
        "seebeck_phrase_idx": center_idx,
        "seebeck_context": text[max(0, center_idx-120): center_idx+120],
        "value_raw": raw,
        "value_numeric": numeric_val,
        "unit": unit,
        "material": mat,
        "material_source": mat_src,
    })

# make dataframe
df = pd.DataFrame(rows)

# Some postprocessing: unit normalization and convert units to microvolt per kelvin (µV/K)
def normalize_unit(u):
    if not u:
        return None
    u = u.lower().replace(" ", "")
    if "uv" in u or "µv" in u or "μv" in u:
        return "uV/K"
    if "mv" in u:
        return "mV/K"
    if "v" in u and "/k" in u:
        return "V/K"
    return u

df["unit_norm"] = df["unit"].apply(normalize_unit)

# convert numeric to a common base (uV/K)
def to_uV_per_K(val, unit):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if unit is None:
        return val
    unit = unit.lower()
    try:
        if "mv" in unit:
            return float(val) * 1000.0
        if ("uv" in unit) or ("µv" in unit) or ("μv" in unit):
            return float(val)
        if "v" in unit and "/k" in unit:
            return float(val) * 1e6
    except Exception:
        return val
    return val

df["value_uV_per_K"] = df.apply(lambda r: to_uV_per_K(r["value_numeric"], r["unit"]), axis=1)

# statistical outlier detection (z-score) on numeric uV/K where available
vals = df["value_uV_per_K"].dropna().astype(float)
if len(vals) >= 2:
    mean = vals.mean()
    std = vals.std(ddof=0) if vals.std(ddof=0) > 0 else 1.0
    df["zscore"] = df["value_uV_per_K"].apply(lambda v: (v - mean)/std if pd.notna(v) else None)
    df["is_outlier"] = df["zscore"].apply(lambda z: abs(z) > outlier_zscore_thresh if z is not None else False)
else:
    df["zscore"] = None
    df["is_outlier"] = False

# show results
st.subheader("Extracted entries (one row per 'Seebeck coefficient' occurrence)")
st.dataframe(df[[
    "seebeck_phrase_idx", "material", "material_source", "value_raw", "value_numeric", "unit", "unit_norm", "value_uV_per_K", "zscore", "is_outlier"
]])

# Save to SQLite db
db_bytes = BytesIO()
conn = sqlite3.connect(":memory:")  # use in-memory then write bytes for download
df.to_sql("seebeck_entries", conn, index=False, if_exists="replace")
# dump DB to bytes
with BytesIO() as b_io:
    for line in conn.iterdump():
        b_io.write(f"{line}\n".encode("utf-8"))
    sql_text = b_io.getvalue()
conn.close()
# Create an actual .db file in-memory using sqlite3 and attach
# We'll create a real .db BytesIO by creating a temporary sqlite file and reading it
tmp_conn = sqlite3.connect("seebeck_data_temp.db")
df.to_sql("seebeck_entries", tmp_conn, index=False, if_exists="replace")
tmp_conn.commit()
tmp_conn.close()
with open("seebeck_data_temp.db", "rb") as f:
    db_data = f.read()

st.download_button("Download SQLite .db", db_data, file_name="seebeck_data.db", mime="application/octet-stream")

# CSV download
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv_bytes, file_name="seebeck_data.csv", mime="text/csv")

# Show summary stats
st.subheader("Summary statistics (numeric values converted to µV/K when unit known)")
if len(vals) > 0:
    st.write(f"Count (numeric): {len(vals)}")
    st.write(f"Mean (µV/K): {mean:.3g}")
    st.write(f"Median (µV/K): {vals.median():.3g}")
    st.write(f"Std (µV/K): {std:.3g}")
    st.write("Outliers flagged:", int(df["is_outlier"].sum()))
else:
    st.write("No numeric Seebeck values parsed.")

st.markdown("**Notes / assumptions:**")
st.markdown("""
- I used a **greedy local search** around each occurrence of the phrase "Seebeck coefficient" and picked the closest numeric+unit match.  
- The code looks for many unit spellings (µV/K, μV/K, uV/K, mV/K, V/K and forms like 'μV K^-1').  
- For material extraction I try to validate chemical formulas with `pymatgen.Composition` if `pymatgen` is installed.  
- If `pymatgen` is not available, the app falls back to heuristics (formula-looking tokens or capitalized names).  
- Statistical outlier detection is a simple z-score method; tune threshold as needed.
""")

if not PYMATGEN_AVAILABLE:
    st.warning("pymatgen not available in this environment — chemical formula validation is disabled. Install via `pip install pymatgen` for more accurate formula parsing.")
