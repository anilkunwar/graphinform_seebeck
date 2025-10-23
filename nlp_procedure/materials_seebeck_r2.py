import streamlit as st
import re
import sqlite3
import pandas as pd
import numpy as np
from io import BytesIO
from math import isnan
import os

# --- PDF Parsers ---
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

# --- Quantity Parsing ---
try:
    from quantulum3 import parser as quant_parser
    QUANTULUM_AVAILABLE = True
except Exception:
    QUANTULUM_AVAILABLE = False

# --- Formula Validation ---
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

# --- NLP ---
try:
    import spacy
    SPACY_AVAILABLE = True
    nlp = spacy.load("en_core_web_sm")
except Exception:
    SPACY_AVAILABLE = False
    nlp = None

# --- Streamlit UI ---
st.title("Advanced PDF → SQLite / CSV: Seebeck Coefficient Extraction")

st.markdown(
    """
    Upload a PDF. The app will extract Seebeck coefficients and materials using improved strategies.
    """
)

# --- Strategy Overview ---
st.subheader("Incorporated Strategies (with Indices)")
st.markdown("""
- **Strategy 1: Enhanced Search Scope and Bidirectionality** – Dynamic window expansion for values; bidirectional material search.  
- **Strategy 2: Improved Linguistic Handling** – Heuristic pronoun resolution, regex-based NER, and multi-tool validation (pymatgen, RDKit, PubChem).  
- **Strategy 3: Robust Unit and Value Parsing** – Quantulum3 integration and improved regex patterns.  
- **Strategy 4: Improved Table Extraction** – Uses Tabula-py for better structured data retrieval.  
- **Strategy 5: Semantic Context Parsing** – SpaCy for contextual extraction and dependency-based sentence linking.  
- **Strategy 6: Evaluation Metrics & Export** – Includes confidence, distance, outlier detection, and SQLite/CSV export.
""")

# --- User Inputs ---
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
            filelike.seek(0)
            with pdfplumber.open(filelike) as pdf:
                for p in pdf.pages:
                    page_tables = p.extract_tables()
                    for t in page_tables:
                        tables.append(pd.DataFrame(t))
        except Exception:
            pass
    elif PYPDF2_AVAILABLE:
        try:
            reader = PdfReader(filelike)
            for page in reader.pages:
                t = page.extract_text() or ""
                text += t + "\n"
        except Exception:
            pass
    if TABULA_AVAILABLE:
        try:
            dfs = tabula.read_pdf(filelike, pages="all", multiple_tables=True)
            tables.extend(dfs)
        except Exception:
            pass

    # Try to extract tables mentioning Seebeck
    table_text = ""
    for df in tables:
        for col in df.columns:
            if df[col].dtype == 'object':
                matches = df[col].str.contains("seebeck coefficient", case=False, na=False)
                if matches.any():
                    table_text += df[matches].to_string() + "\n"

    text += "\n" + table_text

    # Fallback: direct binary read
    if not text.strip():
        try:
            filelike.seek(0)
            text = filelike.read().decode('utf-8', errors='ignore')
        except Exception:
            text = ""
    return text

text = extract_text_from_pdf(uploaded_file)

if len(text.strip()) == 0:
    st.error("Failed to extract text from PDF.")
    st.stop()

st.subheader("Preview (first 2000 chars)")
st.text_area("Extracted text preview", text[:2000], height=220)

# --- Strategy 3: Regex Definitions ---
num_pattern = r"(?P<number>[+\-]?(?:\d{1,3}(?:[,\s]\d{3})*|\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?)"
units_pattern = r"(?:µV\/K|μV\/K|uV\/K|mV\/K|V\/K|μV K(?:⁻¹|-1)?|uV K(?:⁻¹|-1)?|mV K(?:⁻¹|-1)?|V K(?:⁻¹|-1)?|µV⋅K⁻¹)"
dash_variants = r"(?:to|\-|–|—|−)"
value_regex = re.compile(
    rf"(?:(?:±|\+\/\-)\s*)?{num_pattern}(?:\s*{dash_variants}\s*{num_pattern})?\s*(?P<unit>{units_pattern})",
    flags=re.IGNORECASE | re.UNICODE
)

material_regex = re.compile(
    r"\b([A-Z][a-z]?(?:_\{?[a-z0-9\-x]+\}?)?\d*(?:[A-Z][a-z]?(?:_\{?[a-z0-9\-x]+\}?)?\d*)*|[A-Z][a-z]?\d*)\b",
    flags=re.IGNORECASE
)

# --- Strategy 5: SpaCy Sentence Context ---
sc_occurrences = [m for m in re.finditer(r"seebeck coefficient|thermoelectric power|S coefficient", text, flags=re.IGNORECASE)]
st.write(f"Found **{len(sc_occurrences)}** occurrences (including synonyms).")

if SPACY_AVAILABLE and use_spacy:
    doc = nlp(text)
    sentences = list(doc.sents)
else:
    sentences = re.split(r"[\.!?]\s+", text)

def get_context(center_idx):
    if SPACY_AVAILABLE and use_spacy:
        for sent in sentences:
            if sent.start_char <= center_idx < sent.end_char:
                idx = sentences.index(sent)
                para_start = max(0, idx - 1)
                para = " ".join([s.text for s in sentences[para_start:para_start + 3]])
                return para, sent.text
        return text[max(0, center_idx - 200):center_idx + 200], ""
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
                    abs_start = start + snippet.find(str(q.value))
                    dist = abs(center_idx - abs_start)
                    matches.append({
                        "abs_start": abs_start,
                        "abs_end": abs_start + len(str(q.value)),
                        "dist": dist,
                        "raw": f"{q.value} {q.unit.name}",
                        "numeric": q.value,
                        "unit": q.unit.name,
                    })
        else:
            for m in value_regex.finditer(snippet):
                abs_start = start + m.start()
                abs_end = start + m.end()
                dist = abs(center_idx - (abs_start + abs_end)//2)
                raw = m.group(0).strip()
                number_text = m.group("number") or re.search(r"[+\-]?\d+\.?\d*", raw).group(0)
                number_text_norm = number_text.replace(",", "").replace(" ", "")
                try:
                    numeric = float(number_text_norm)
                except:
                    numeric = None
                unit = m.group("unit")
                matches.append({
                    "abs_start": abs_start,
                    "abs_end": abs_end,
                    "dist": dist,
                    "raw": raw,
                    "numeric": numeric,
                    "unit": unit,
                })
        if matches:
            return sorted(matches, key=lambda x: x["dist"])
        window *= 2
    return []

# --- Strategy 2: Material Search ---
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
            if len(cand.split()) <= max_tokens:
                candidates.append(cand)
        if SPACY_AVAILABLE and use_spacy:
            ctx_doc = nlp(ctx)
            for chunk in ctx_doc.noun_chunks:
                if re.search(r"[A-Z][a-z]{2,}", chunk.text) or material_regex.search(chunk.text):
                    if len(chunk.text.split()) <= max_tokens:
                        candidates.append(chunk.text)
    for c in set(candidates):
        c_clean = re.sub(r"[_}{]", "", c).strip()
        if PYMATGEN_AVAILABLE:
            try:
                comp = Composition(c_clean)
                if sum(comp.get_el_amt_dict().values()) > 0:
                    prev_mat = c_clean
                    return c_clean, "validated_pymatgen"
            except:
                pass
        if RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(c_clean)
                if mol:
                    prev_mat = c_clean
                    return c_clean, "validated_rdkit"
            except:
                pass
        if PUBCHEM_AVAILABLE:
            try:
                compounds = pcp.get_compounds(c_clean, 'formula')
                if compounds:
                    prev_mat = c_clean
                    return c_clean, "validated_pubchem"
            except:
                pass
        if material_regex.match(c_clean):
            prev_mat = c_clean
            return c_clean, "heuristic"
    return "", "none_found"

# --- Process Occurrences ---
rows = []
for occ in sc_occurrences:
    center_idx = occ.start()
    context, sentence = get_context(center_idx)
    candidates = find_nearest_value(center_idx, context if context else text,
                                    initial_window=initial_window_chars, max_window=max_window_chars)
    if not candidates:
        rows.append({
            "seebeck_phrase_idx": center_idx,
            "seebeck_context": sentence or text[max(0, center_idx-120):center_idx+120],
            "value_raw": None,
            "value_numeric": None,
            "unit": None,
            "material": None,
            "material_source": "no_value_found",
            "dist": None,
            "confidence": 0.0,
        })
        continue
    top = candidates[0]
    mat, mat_src = find_material_around(top["abs_start"], top["abs_end"], text, max_tokens=max_material_tokens)
    rows.append({
        "seebeck_phrase_idx": center_idx,
        "seebeck_context": sentence or text[max(0, center_idx-120):center_idx+120],
        "value_raw": top["raw"],
        "value_numeric": top.get("numeric"),
        "unit": top.get("unit"),
        "material": mat,
        "material_source": mat_src,
        "dist": top["dist"],
        "confidence": 1 / (1 + top["dist"]/100.0),
    })

df = pd.DataFrame(rows)

# --- Unit Normalization ---
def normalize_unit(u):
    if not u:
        return None
    u = u.lower().replace(" ", "").replace("⋅", "/").replace("-1", "^-1")
    if any(x in u for x in ["uv", "µv", "μv", "microvolt"]):
        return "uV/K"
    if "mv" in u:
        return "mV/K"
    if "v/k" in u:
        return "V/K"
    return u

df["unit_norm"] = df["unit"].apply(normalize_unit)

def to_uV_per_K(val, unit):
    if pd.isna(val):
        return None
    factor = 1
    if unit and "mv" in unit.lower():
        factor = 1000
    elif unit and "v/" in unit.lower():
        factor = 1e6
    return float(val) * factor

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
st.dataframe(df[[
    "seebeck_phrase_idx", "material", "material_source", "value_raw", "value_numeric",
    "unit", "unit_norm", "value_uV_per_K", "zscore", "is_outlier", "dist", "confidence"
]])

# --- Effectiveness Indices ---
if not df.empty:
    st.subheader("Effectiveness Indices")
    st.write(f"**Value Extraction Success Rate:** {(df['value_numeric'].notna().sum()/len(df))*100:.2f}%")
    st.write(f"**Material Association Rate:** {(df['material'] != '').sum()/len(df)*100:.2f}%")
    st.write(f"**Average Value Distance:** {df['dist'].dropna().mean():.2f} chars")
    st.write(f"**Average Confidence Score:** {df['confidence'].mean():.2f}")
    st.write(f"**Outlier Rate:** {(df['is_outlier'].sum()/len(df))*100:.2f}%")
    st.write(f"**Unique Units Parsed:** {df['unit_norm'].nunique()}")

# --- SQLite Export ---
temp_db_path = "seebeck_data_temp.db"
temp_conn = sqlite3.connect(temp_db_path)
df.to_sql("seebeck_entries", temp_conn, index=False, if_exists="replace")
temp_conn.commit()
temp_conn.close()
with open(temp_db_path, "rb") as f:
    db_data = f.read()
os.remove(temp_db_path)
st.download_button("Download SQLite .db", db_data, file_name="seebeck_data.db", mime="application/octet-stream")

# --- CSV Export ---
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv_bytes, file_name="seebeck_data.csv", mime="text/csv")

# --- Summary ---
st.subheader("Summary Statistics (µV/K)")
if len(vals) > 0:
    st.write(f"Count: {len(vals)}")
    st.write(f"Mean: {vals.mean():.3g} µV/K")
    st.write(f"Median: {vals.median():.3g} µV/K")
    st.write(f"Std: {vals.std(ddof=0):.3g}")
    st.write("Outliers flagged:", int(df["is_outlier"].sum()))
else:
    st.write("No numeric Seebeck values parsed.")

# --- Warnings ---
if not PYMATGEN_AVAILABLE:
    st.warning("pymatgen not available — limited formula validation.")
if not SPACY_AVAILABLE:
    st.warning("spaCy not available — no semantic parsing.")
if not QUANTULUM_AVAILABLE:
    st.warning("quantulum3 not available — regex-based quantity extraction only.")
if not TABULA_AVAILABLE:
    st.warning("tabula-py not available — limited table extraction.")
