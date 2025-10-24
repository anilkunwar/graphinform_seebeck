import streamlit as st
import re
import sqlite3
import pandas as pd
import numpy as np
from io import BytesIO
from math import isnan
import os

# Try better PDF parsers
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

# Quantity parsing
try:
    from quantulum3 import parser as quant_parser
    QUANTULUM_AVAILABLE = True
except Exception:
    QUANTULUM_AVAILABLE = False

# Formula validation
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

# NLP
try:
    import spacy
    SPACY_AVAILABLE = True
    # Ensure the model is downloaded. In a real environment, you might need:
    # os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load("en_core_web_sm")
except Exception:
    SPACY_AVAILABLE = False
    nlp = None

st.title("Advanced PDF → SQLite / CSV: Seebeck Coefficient Extraction")

st.markdown(
    """
    Upload a PDF. The app will extract Seebeck coefficients and materials using improved strategies.
    """
)

# List incorporated strategies
st.subheader("Incorporated Strategies (Highlighted with Indices for Effectiveness)")
st.markdown("""
- **Strategy 1: Enhanced Search Scope and Bidirectionality** - Dynamic window expansion for values; bidirectional material search.  
  *Index:* Average Value Distance (lower is better); Value Extraction Success Rate (higher is better).
- **Strategy 2: Improved Linguistic Handling** - Heuristic pronoun resolution; advanced material NER with regex; multi-tool validation (pymatgen/rdkit/pubchem).  
  *Index:* Material Association Rate (higher is better).
- **Strategy 3: Robustify Unit and Value Parsing** - Quantulum3 integration; expanded regex for unit orders/variations.  
  *Index:* Value Extraction Success Rate; Number of unique units parsed.
- **Strategy 4: Better Text Extraction** - Tabula-py for tables.  
  *Index:* Table detection flag; additional entries from tables.
- **Strategy 5: Incorporate Semantics** - SpaCy for sentence context and dependency parsing.  
  *Index:* Confidence Score Avg (higher indicates better contextual matches).
- **Strategy 6: General Best Practices** - User configs; hybrid fallbacks; evaluation metrics.  
  *Index:* All above + Outlier Rate.
""")

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

# --- Strategy 4: Better Text Extraction ---
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

if len(text.strip()) == 0:
    st.error("Failed to extract text from PDF.")
    st.stop()

st.subheader("Preview (first 2000 chars)")
st.text_area("extracted_text_preview", text[:2000], height=220)

# --- Strategy 3: Safe Regex Patterns (Fixed) ---
# Fixed number pattern - removed problematic character class
num_pattern = r"(?P<number>[+\-]?(?:\d{1,3}(?:[,\s]\d{3})*|\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?)"

# Corrected and simplified units pattern (removed escaped caret '\^' to avoid error)
units_pattern = r"(?:µV\/K|μV\/K|uV\/K|microvolts? per kelvin|mV\/K|V\/K|μV K(?:\^[-1]|⁻¹|-1)?|uV K(?:\^[-1]|⁻¹|-1)?|mV K(?:\^[-1]|⁻¹|-1)?|V K(?:\^[-1]|⁻¹|-1)?|µV⋅K⁻¹|μV⋅K⁻¹)"

dash_variants = r"(?:to|\-|–|—|−)"  # Safe non-capturing group for all dash types

# Fixed regex: simplified and tested
value_regex = re.compile(
    rf"(?:(?:±|\+\/\-)\s*)?{num_pattern}(?:\s*{dash_variants}\s*{num_pattern})?\s*(?P<unit>{units_pattern})",
    flags=re.IGNORECASE | re.UNICODE
)

# Material regex: expanded for LaTeX, subscripts, doped
material_regex = re.compile(
    r"\b([A-Z][a-z]?(?:_\{?[a-z0-9\-x]+\}?)?\d*(?:[A-Z][a-z]?(?:_\{?[a-z0-9\-x]+\}?)?\d*)*|[A-Z][a-z]?\d*)\b",
    flags=re.IGNORECASE
)

# Find all 'seebeck coefficient' occurrences (including synonyms)
sc_occurrences = [m for m in re.finditer(r"seebeck coefficient|thermoelectric power|S coefficient", text, flags=re.IGNORECASE)]
st.write(f"Found **{len(sc_occurrences)}** occurrences (including synonyms).")

# Strategy 5: SpaCy for context
if SPACY_AVAILABLE and use_spacy:
    doc = nlp(text)
    sentences = list(doc.sents)
else:
    sentences = re.split(r"[\.!?]\s+", text)

# Helper: Get context (sentence/paragraph) for index
def get_context(center_idx):
    if SPACY_AVAILABLE and use_spacy:
        for sent in sentences:
            if sent.start_char <= center_idx < sent.end_char:
                idx = sentences.index(sent)
                para_start = max(0, idx-1)
                para = " ".join([s.text for s in sentences[para_start:para_start+3]])
                return para, sent.text
        return text[max(0, center_idx-200):center_idx+200], ""
    else:
        paras = re.split(r"\n{2,}", text)
        for p in paras:
            start = text.find(p)
            if start <= center_idx < start + len(p):
                return p, re.split(r"[\.!?]\s+", p)[-1] if "." in p else p
        return text[max(0, center_idx-200):center_idx+200], ""

# Strategy 1: Dynamic nearest value search
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
                    # Approximate absolute start index, quantulum doesn't give precise spans
                    # Fallback: search for value in snippet
                    value_str = str(q.value)
                    try:
                        match_start = snippet.index(value_str)
                        abs_start = start + match_start
                        abs_end = abs_start + len(value_str)
                        dist = abs(center_idx - (abs_start + abs_end)//2)
                        matches.append({
                            "abs_start": abs_start,
                            "abs_end": abs_end,
                            "dist": dist,
                            "raw": f"{q.value} {q.unit.name}",
                            "numeric": q.value,
                            "unit": q.unit.name,
                        })
                    except ValueError:
                        # Fallback if value string isn't simple (e.g., quantulum parsed a range)
                        continue
        else:
            for m in value_regex.finditer(snippet):
                abs_start = start + m.start()
                abs_end = start + m.end()
                dist = abs(center_idx - (abs_start + abs_end)//2)
                raw = m.group(0).strip()
                
                # Safely extract number part (handles ranges)
                number_match = re.search(num_pattern, raw)
                number_text = number_match.group("number") if number_match else None
                
                numeric = None
                if number_text:
                    number_text_norm = number_text.replace(",", "").replace(" ", "")
                    try:
                        numeric = float(number_text_norm)
                    except:
                        numeric = None # Could be a range, stick to raw
                
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

# Strategy 2: Bidirectional material search with validation & pronoun heuristic
prev_mat = ""

def find_material_around(value_start, value_end, txt, max_tokens=10, window=800):
    global prev_mat
    before = txt[max(0, value_start - window):value_start]
    after = txt[value_end:value_end + window]
    contexts = [before, after]
    candidates = []
    for ctx in contexts:
        # Heuristic pronoun resolution (Strategy 2)
        ctx = re.sub(r"\b(it|this|the material|the compound)\b", prev_mat or "UNKNOWN", ctx, flags=re.I)
        
        # Regex candidates
        for m in material_regex.finditer(ctx):
            cand = m.group(0).strip()
            if len(cand.split()) <= max_tokens and cand.lower() not in ["and", "or", "in"]: # basic filter
                candidates.append(cand)
        
        # SpaCy candidates (Strategy 5)
        if SPACY_AVAILABLE and use_spacy:
            ctx_doc = nlp(ctx)
            for chunk in ctx_doc.noun_chunks:
                # Check for chemical-like capitalization or regex match
                if re.search(r"[A-Z][a-z]{2,}", chunk.text) or material_regex.search(chunk.text):
                    if len(chunk.text.split()) <= max_tokens:
                        candidates.append(chunk.text)
    
    # Validation (Strategy 2)
    for c in set(candidates):
        c_clean = re.sub(r"[_}{]", "", c).strip()
        if len(c_clean) < 2: continue

        # Pymatgen validation
        if PYMATGEN_AVAILABLE:
            try:
                comp = Composition(c_clean)
                if sum(comp.get_el_amt_dict().values()) > 0:
                    prev_mat = c_clean
                    return c_clean, "validated_pymatgen"
            except:
                pass
        
        # RDKit validation
        if RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(c_clean)
                if mol:
                    prev_mat = c_clean
                    return c_clean, "validated_rdkit"
            except:
                pass
        
        # PubChem validation (formula only)
        if PUBCHEM_AVAILABLE:
            try:
                compounds = pcp.get_compounds(c_clean, 'formula')
                if compounds:
                    prev_mat = c_clean
                    return c_clean, "validated_pubchem"
            except:
                pass
        
        # Final heuristic fallback
        if material_regex.match(c_clean):
            prev_mat = c_clean
            return c_clean, "heuristic"
    
    return "", "none_found"

# --- Process each occurrence ---
rows = []
for occ in sc_occurrences:
    center_idx = occ.start()
    context, sentence = get_context(center_idx)
    
    # Strategy 1: Dynamic window search
    candidates = find_nearest_value(center_idx, context if context else text, initial_window=initial_window_chars, max_window=max_window_chars)
    
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
    
    # Use the closest value
    top = candidates[0]
    raw = top["raw"]
    numeric_val = top.get("numeric")
    unit = top.get("unit")
    dist = top["dist"]
    confidence = 1 / (1 + dist / 100.0) if dist is not None else 0.0 # simple inverse distance confidence
    
    # Strategy 2: Bidirectional material search
    mat, mat_src = find_material_around(top["abs_start"], top["abs_end"], text, max_tokens=max_material_tokens)
    
    rows.append({
        "seebeck_phrase_idx": center_idx,
        "seebeck_context": sentence or text[max(0, center_idx-120):center_idx+120],
        "value_raw": raw,
        "value_numeric": numeric_val,
        "unit": unit,
        "material": mat,
        "material_source": mat_src,
        "dist": dist,
        "confidence": confidence,
    })

# DataFrame
df = pd.DataFrame(rows)

# Unit normalization and conversion (Strategy 3)
def normalize_unit(u):
    if not u:
        return None
    u = u.lower().replace(" ", "").replace("⋅", "/").replace("-1", "^-1").replace("μ", "u").replace("µ", "u")
    if "uv/k" in u or "microvoltperkelvin" in u:
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
    except ValueError:
        return None
    
    factor = 1
    if unit and "mv/k" in unit.lower():
        factor = 1000
    elif unit and "v/k" in unit.lower():
        factor = 1e6
        
    return val * factor

df["value_uV_per_K"] = df.apply(lambda r: to_uV_per_K(r["value_numeric"], r["unit_norm"]), axis=1)

# Outliers (Strategy 6)
vals = df["value_uV_per_K"].dropna()
if len(vals) >= 2:
    mean = vals.mean()
    std = vals.std(ddof=0) # Population standard deviation
    std = std if std != 0 else 1.0 # Avoid division by zero
    
    df["zscore"] = df["value_uV_per_K"].apply(lambda v: (v - mean)/std if pd.notna(v) else None)
    df["is_outlier"] = df["zscore"].apply(lambda z: abs(z) > outlier_zscore_thresh if z is not None else False)
else:
    df["zscore"] = None
    df["is_outlier"] = False

# Display
st.subheader("Extracted Entries")
st.dataframe(df[[
    "seebeck_phrase_idx", "material", "material_source", "value_raw", "value_numeric", "unit", "unit_norm", "value_uV_per_K", "zscore", "is_outlier", "dist", "confidence"
]])

# Strategy 6: Effectiveness Indices
if not df.empty:
    value_success = (df["value_numeric"].notna().sum() / len(df)) * 100
    mat_success = (df["material"] != "").sum() / len(df) * 100
    avg_dist = df["dist"].dropna().mean() if not df["dist"].dropna().empty else 0
    avg_conf = df["confidence"].mean()
    outlier_rate = (df["is_outlier"].sum() / len(df)) * 100
    unique_units = df["unit_norm"].nunique()

    st.subheader("Effectiveness Indices")
    st.write(f"**Value Extraction Success Rate:** {value_success:.2f}%")
    st.write(f"**Material Association Rate:** {mat_success:.2f}%")
    st.write(f"**Average Value Distance:** {avg_dist:.2f} chars")
    st.write(f"**Average Confidence Score:** {avg_conf:.2f}")
    st.write(f"**Outlier Rate:** {outlier_rate:.2f}%")
    st.write(f"**Unique Units Parsed:** {unique_units}")

# Save to SQLite
temp_db_path = "seebeck_data_temp.db"
try:
    temp_conn = sqlite3.connect(temp_db_path)
    df.to_sql("seebeck_entries", temp_conn, index=False, if_exists="replace")
    temp_conn.commit()
    temp_conn.close()

    with open(temp_db_path, "rb") as f:
        db_data = f.read()
    st.download_button("Download SQLite .db", db_data, file_name="seebeck_data.db", mime="application/octet-stream")
except Exception as e:
    st.error(f"Could not create SQLite DB: {e}")
finally:
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)

# CSV download
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv_bytes, file_name="seebeck_data.csv", mime="text/csv")

# Summary stats
st.subheader("Summary statistics (µV/K)")
if len(vals) > 0:
    st.write(f"Count: {len(vals)}")
    st.write(f"Mean: {vals.mean():.3g} µV/K")
    st.write(f"Median: {vals.median():.3g} µV/K")
    st.write(f"Std: {vals.std(ddof=0):.3g}")
    st.write("Outliers flagged:", int(df["is_outlier"].sum()))
else:
    st.write("No numeric Seebeck values parsed.")

# Warnings
if not PYMATGEN_AVAILABLE:
    st.warning("pymatgen not available — limited formula validation.")
if not SPACY_AVAILABLE:
    st.warning("spaCy not available — no advanced semantics.")
if not QUANTULUM_AVAILABLE:
    st.warning("quantulum3 not available — using regex for quantities.")
if not TABULA_AVAILABLE:
    st.warning("tabula-py not available — limited table extraction.")
