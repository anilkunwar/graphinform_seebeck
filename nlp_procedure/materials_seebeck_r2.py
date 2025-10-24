import streamlit as st
import re
import sqlite3
import pandas as pd
import numpy as np
from io import BytesIO
from math import isnan
import os

# --- Optional Imports ---
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

try:
    import tabula
    TABULA_AVAILABLE = True
except Exception:
    TABULA_AVAILABLE = False

try:
    from quantulum3 import parser as quant_parser
    QUANTULUM_AVAILABLE = True
except Exception:
    QUANTULUM_AVAILABLE = False

try:
    from pymatgen.core.composition import Composition
    PYMATGEN_AVAILABLE = True
except Exception:
    PYMATGEN_AVAILABLE = False

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False

try:
    import pubchempy as pcp
    PUBCHEM_AVAILABLE = True
except Exception:
    PUBCHEM_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
    nlp = spacy.load("en_core_web_sm")
except Exception:
    SPACY_AVAILABLE = False
    nlp = None

# --- UI ---
st.title("Advanced PDF → SQLite / CSV: Seebeck Coefficient Extraction")

st.markdown("""
Upload a PDF. The app will extract Seebeck coefficients and materials using improved strategies.
""")

# --- Strategy Overview ---
st.subheader("Incorporated Strategies")
st.markdown("""
- **Strategy 1:** Dynamic window expansion and bidirectional material search  
- **Strategy 2:** Heuristic pronoun resolution and validation (pymatgen, RDKit, PubChem)  
- **Strategy 3:** Robust regex and unit normalization  
- **Strategy 4:** Improved table and text extraction  
- **Strategy 5:** Context understanding via spaCy (if available)  
- **Strategy 6:** Outlier filtering and effectiveness metrics  
""")

# --- Streamlit Inputs ---
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
initial_window_chars = st.slider("Initial context window (chars)", 200, 2000, 600, 10)
max_window_chars = st.slider("Max context window (chars)", 1000, 10000, 5000, 100)
max_material_tokens = st.slider("Max tokens for material name", 1, 20, 10)
outlier_zscore_thresh = st.number_input("Z-score threshold for outliers", 0.0, 10.0, 2.5, 0.1)
use_spacy = st.checkbox("Use spaCy for semantics (if available)", value=SPACY_AVAILABLE)
use_quantulum = st.checkbox("Use quantulum3 for quantities (if available)", value=QUANTULUM_AVAILABLE)

if uploaded_file is None:
    st.info("Upload a PDF to start.")
    st.stop()

# --- Strategy 4: Text Extraction ---
def extract_text_from_pdf(filelike):
    text = ""
    tables = []
    filelike.seek(0)

    if PDF_PLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(filelike) as pdf:
                for p in pdf.pages:
                    page_text = p.extract_text()
                    if page_text:
                        text += page_text + "\n"
                for p in pdf.pages:
                    for t in p.extract_tables():
                        if t:
                            tables.append(pd.DataFrame(t))
        except Exception:
            pass

    elif PYPDF2_AVAILABLE:
        try:
            filelike.seek(0)
            reader = PdfReader(filelike)
            for page in reader.pages:
                t = page.extract_text() or ""
                text += t + "\n"
        except Exception:
            pass

    if TABULA_AVAILABLE:
        try:
            filelike.seek(0)
            dfs = tabula.read_pdf(filelike, pages="all", multiple_tables=True)
            tables.extend(dfs)
        except Exception:
            pass

    # Convert tables mentioning "Seebeck" to text
    table_text = ""
    for df in tables:
        for col in df.columns:
            if df[col].dtype == 'object':
                matches = df[col].str.contains("seebeck coefficient", case=False, na=False)
                if matches.any():
                    table_text += df[matches].to_string() + "\n"
    text += "\n" + table_text

    if not text.strip():
        try:
            filelike.seek(0)
            text = filelike.read().decode('utf-8', errors='ignore')
        except Exception:
            text = ""
    return text

text = extract_text_from_pdf(uploaded_file)

if not text.strip():
    st.error("Failed to extract text from PDF.")
    st.stop()

st.subheader("Preview (first 2000 chars)")
st.text_area("Extracted Text Preview", text[:2000], height=220)

# --- Strategy 3: Regex Patterns (Corrected) ---
num_pattern = r"(?P<number>[+\-]?\d+(?:[.,]\d+)?(?:[eE][+\-]?\d+)?)"
units_pattern = r"(?:µV\/K|μV\/K|uV\/K|microvolt[s]? per kelvin|mV\/K|V\/K|μV K(?:\^-1|⁻¹|-1)?|uV K(?:\^-1|⁻¹|-1)?|mV K(?:\^-1|⁻¹|-1)?|V K(?:\^-1|⁻¹|-1)?|µV⋅K⁻¹|μV⋅K⁻¹)"
dash_variants = r"(?:to|−|–|—|-)"  # handles multiple dash types safely

value_regex = re.compile(
    rf"(?:(?:±|\+/-)\s*)?(?:{num_pattern})(?:\s*{dash_variants}\s*{num_pattern})?\s*(?P<unit>{units_pattern})",
    flags=re.IGNORECASE | re.UNICODE
)

material_regex = re.compile(
    r"\b([A-Z][a-z]?(?:_\{?[a-z0-9\-x]+\}?)?\d*(?:[A-Z][a-z]?(?:_\{?[a-z0-9\-x]+\}?)?\d*)*|[A-Z][a-z]?\d*)\b",
    flags=re.IGNORECASE
)

# --- Find Seebeck Mentions ---
sc_occurrences = [m for m in re.finditer(r"seebeck coefficient|thermoelectric power|S coefficient", text, flags=re.IGNORECASE)]
st.write(f"Found **{len(sc_occurrences)}** relevant phrases.")

# --- spaCy Sentence Split ---
if SPACY_AVAILABLE and use_spacy:
    doc = nlp(text)
    sentences = list(doc.sents)
else:
    sentences = re.split(r"[\.!?]\s+", text)

# --- Context Helper ---
def get_context(center_idx):
    if SPACY_AVAILABLE and use_spacy:
        for sent in sentences:
            if sent.start_char <= center_idx < sent.end_char:
                idx = sentences.index(sent)
                para_start = max(0, idx - 1)
                para = " ".join([s.text for s in sentences[para_start:para_start + 3]])
                return para, sent.text
    else:
        paras = re.split(r"\n{2,}", text)
        for p in paras:
            start = text.find(p)
            if start <= center_idx < start + len(p):
                return p, re.split(r"[\.!?]\s+", p)[-1] if "." in p else p
    return text[max(0, center_idx - 200):center_idx + 200], ""

# --- Strategy 1: Value Finder ---
def find_nearest_value(center_idx, txt, initial_window=600, max_window=5000):
    window = initial_window
    while window <= max_window:
        start = max(0, center_idx - window)
        end = min(len(txt), center_idx + window)
        snippet = txt[start:end]
        matches = []

        if QUANTULUM_AVAILABLE and use_quantulum:
            quants = quant_parser.parse(snippet)
            for q in quants:
                if "volt" in q.unit.name.lower() and "kelvin" in q.unit.name.lower():
                    try:
                        match_start = snippet.index(str(q.value))
                        abs_start = start + match_start
                        abs_end = abs_start + len(str(q.value))
                        dist = abs(center_idx - (abs_start + abs_end) // 2)
                        matches.append({
                            "abs_start": abs_start,
                            "abs_end": abs_end,
                            "dist": dist,
                            "raw": f"{q.value} {q.unit.name}",
                            "numeric": q.value,
                            "unit": q.unit.name,
                        })
                    except ValueError:
                        continue
        else:
            for m in value_regex.finditer(snippet):
                abs_start = start + m.start()
                abs_end = start + m.end()
                dist = abs(center_idx - (abs_start + abs_end) // 2)
                raw = m.group(0).strip()
                number_match = re.search(num_pattern, raw)
                numeric = None
                if number_match:
                    try:
                        numeric = float(number_match.group("number").replace(",", "").replace(" ", ""))
                    except Exception:
                        pass
                matches.append({
                    "abs_start": abs_start,
                    "abs_end": abs_end,
                    "dist": dist,
                    "raw": raw,
                    "numeric": numeric,
                    "unit": m.group("unit"),
                })
        if matches:
            return sorted(matches, key=lambda x: x["dist"])
        window *= 2
    return []

# --- Strategy 2: Material Finder ---
prev_mat = ""
def find_material_around(value_start, value_end, txt, max_tokens=10, window=800):
    global prev_mat
    before = txt[max(0, value_start - window):value_start]
    after = txt[value_end:value_end + window]
    contexts = [before, after]
    candidates = []

    for ctx in contexts:
        ctx = re.sub(r"\b(it|this|the material|the compound)\b", prev_mat or "UNKNOWN", ctx, flags=re.I)
        for m in material_regex.finditer(ctx):
            cand = m.group(0).strip()
            if len(cand.split()) <= max_tokens and cand.lower() not in ["and", "or", "in"]:
                candidates.append(cand)

    for c in set(candidates):
        c_clean = re.sub(r"[_}{]", "", c).strip()
        if len(c_clean) < 2:
            continue
        if PYMATGEN_AVAILABLE:
            try:
                comp = Composition(c_clean)
                if sum(comp.get_el_amt_dict().values()) > 0:
                    prev_mat = c_clean
                    return c_clean, "validated_pymatgen"
            except Exception:
                pass
        if material_regex.match(c_clean):
            prev_mat = c_clean
            return c_clean, "heuristic"
    return "", "none_found"

# --- Process Each Occurrence ---
rows = []
for occ in sc_occurrences:
    center_idx = occ.start()
    context, sentence = get_context(center_idx)
    candidates = find_nearest_value(center_idx, context if context else text, initial_window_chars, max_window_chars)
    if not candidates:
        continue
    top = candidates[0]
    mat, mat_src = find_material_around(top["abs_start"], top["abs_end"], text, max_material_tokens)
    rows.append({
        "seebeck_phrase_idx": center_idx,
        "seebeck_context": sentence,
        "value_raw": top["raw"],
        "value_numeric": top["numeric"],
        "unit": top["unit"],
        "material": mat,
        "material_source": mat_src,
        "dist": top["dist"],
        "confidence": 1 / (1 + top["dist"] / 100.0),
    })

# --- DataFrame ---
df = pd.DataFrame(rows)
if df.empty:
    st.warning("No Seebeck values found.")
    st.stop()

# --- Normalize Units ---
def normalize_unit(u):
    if not u:
        return None
    u = u.lower().replace(" ", "").replace("⋅", "/").replace("-1", "^-1").replace("μ", "u").replace("µ", "u")
    if "uv/k" in u:
        return "uV/K"
    if "mv/k" in u:
        return "mV/K"
    if "v/k" in u:
        return "V/K"
    return u

df["unit_norm"] = df["unit"].apply(normalize_unit)

def to_uV_per_K(val, unit):
    if pd.isna(val) or val is None:
        return None
    try:
        val = float(val)
    except Exception:
        return None
    if not unit:
        return val
    if "mv/k" in unit.lower():
        return val * 1000
    if "v/k" in unit.lower():
        return val * 1e6
    return val

df["value_uV_per_K"] = df.apply(lambda r: to_uV_per_K(r["value_numeric"], r["unit_norm"]), axis=1)

# --- Outlier Detection ---
vals = df["value_uV_per_K"].dropna()
if len(vals) >= 2:
    mean = vals.mean()
    std = vals.std(ddof=0) or 1.0
    df["zscore"] = df["value_uV_per_K"].apply(lambda v: (v - mean)/std if pd.notna(v) else None)
    df["is_outlier"] = df["zscore"].apply(lambda z: abs(z) > outlier_zscore_thresh if z is not None else False)
else:
    df["zscore"] = None
    df["is_outlier"] = False

# --- Display ---
st.subheader("Extracted Entries")
st.dataframe(df)

# --- Download Options ---
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv_bytes, file_name="seebeck_data.csv", mime="text/csv")

# --- Effectiveness Indices ---
if not df.empty:
    st.subheader("Effectiveness Indices")
    st.write(f"**Value Extraction Success Rate:** {(df['value_numeric'].notna().mean() * 100):.2f}%")
    st.write(f"**Material Association Rate:** {(df['material'] != '').mean() * 100:.2f}%")
    st.write(f"**Average Confidence:** {df['confidence'].mean():.3f}")
    st.write(f"**Outlier Rate:** {(df['is_outlier'].mean() * 100):.2f}%")

# --- Warnings ---
if not PYMATGEN_AVAILABLE:
    st.warning("pymatgen not available — limited formula validation.")
if not SPACY_AVAILABLE:
    st.warning("spaCy not available — no semantic parsing.")
if not QUANTULUM_AVAILABLE:
    st.warning("quantulum3 not available — using regex for quantities.")
if not TABULA_AVAILABLE:
    st.warning("tabula-py not available — limited table extraction.")
