# app.py -- Upgraded Seebeck extraction pipeline
# Install suggestions (optional for full features):
# pip install streamlit pdfplumber PyPDF2 pymatgen quantulum3 spacy chemdataextractor camelot-py[cv] tabula-py transformers
# python -m spacy download en_core_web_sm

import streamlit as st
import re
import sqlite3
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
import tempfile
import os
import math
from math import isnan

# --- Optional libraries (graceful fallback) ---
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
    from pymatgen.core.composition import Composition
    PYMATGEN_AVAILABLE = True
except Exception:
    PYMATGEN_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
    # load small english model if possible (deferred)
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        # user may not have model installed
        nlp = None
except Exception:
    SPACY_AVAILABLE = False
    nlp = None

try:
    from quantulum3 import parser as quant_parser
    QUANTULUM_AVAILABLE = True
except Exception:
    QUANTULUM_AVAILABLE = False

# ChemDataExtractor for chemical entity extraction
try:
    from chemdataextractor import Document as CDE_Document
    from chemdataextractor.doc import Paragraph
    CHEMDATA_AVAILABLE = True
except Exception:
    CHEMDATA_AVAILABLE = False

# Table extraction (camelot or tabula)
try:
    import camelot
    CAMELOT_AVAILABLE = True
except Exception:
    CAMELOT_AVAILABLE = False

try:
    import tabula
    TABULA_AVAILABLE = True
except Exception:
    TABULA_AVAILABLE = False

# Coreference resolution (transformers pipeline)
try:
    from transformers import pipeline as hf_pipeline
    COREF_AVAILABLE = True
except Exception:
    COREF_AVAILABLE = False

# Streamlit UI
st.set_page_config(layout="wide")
st.title("PDF → SQLite / CSV: Improved Seebeck Extractor (sentence-aware, quantulum3, tables, coref)")

st.markdown("""
Upload a PDF. The app will:
- find occurrences of Seebeck (and synonyms),
- perform sentence-aware and bidirectional numeric+unit search,
- use quantulum3 where available for robust unit parsing,
- attempt to extract material names/formulas with ChemDataExtractor / pymatgen / heuristics,
- search tables using Camelot/Tabula if installed,
- save results to a SQLite `.db` and CSV.
""")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded_file is None:
    st.info("Upload a PDF to start.")
    st.stop()

# user options
expand_sentences = st.checkbox("Expand to neighboring sentences when searching (recommended)", value=True)
use_coref = st.checkbox("Attempt coreference resolution (may be slow / requires transformer model)", value=False)
search_tables = st.checkbox("Search tables in PDF (camelot/tabula if available)", value=True)
outlier_zscore_thresh = st.number_input("Z-score threshold to flag outliers", 0.0, 10.0, 2.5, 0.1)

# --- helper: robust text extraction ---
def extract_text_from_pdf(filelike):
    text = ""
    filelike.seek(0)
    # Save to temp file (some table libs require a path)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(filelike.getbuffer())
    tmp.flush()
    tmp.close()
    path = tmp.name

    if PDF_PLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(path) as pdf:
                for p in pdf.pages:
                    page_text = p.extract_text()
                    if page_text:
                        text += page_text + "\n"
            os.unlink(path)
            return text
        except Exception:
            pass

    if PYPDF2_AVAILABLE:
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                try:
                    t = page.extract_text()
                except Exception:
                    t = ""
                if t:
                    text += t + "\n"
            os.unlink(path)
            return text
        except Exception:
            pass

    # fallback: read bytes and naive decode
    try:
        filelike.seek(0)
        text = filelike.read().decode('utf-8', errors='ignore')
    except Exception:
        text = ""
    os.unlink(path)
    return text

# extract text
text = extract_text_from_pdf(uploaded_file)
if len(text.strip()) == 0:
    st.error("Failed to extract text from PDF (no text found). Try a different PDF or install pdfplumber for better extraction.")
    st.stop()

st.subheader("Preview (first 2000 chars)")
st.text_area("extracted_text_preview", text[:2000], height=200)

# --- pre-normalize text for unit characters (but keep original separately) ---
text_norm = text.replace("µ", "u").replace("μ", "u").replace("−", "-").replace("\u2212", "-")

# --- keywords (expandable) ---
KEYWORDS = r"(seebeck coefficient|thermoelectric power|S-value|S coefficient|Seebeck)"
keyword_re = re.compile(KEYWORDS, flags=re.IGNORECASE)

# --- utility: sentence segmentation (spaCy fallback to naive split) ---
def sentence_segments(txt):
    if SPACY_AVAILABLE and nlp is not None:
        doc = nlp(txt)
        return list(doc.sents)
    else:
        # naive split by sentence punctuation (keeps context)
        parts = re.split(r'(?<=[\.\!\?])\s+', txt)
        return parts

# --- optional coref resolution (transformers) ---
def try_coref_resolution(txt):
    if not COREF_AVAILABLE or not use_coref:
        return txt
    try:
        # load coref model (may be heavy)
        coref = hf_pipeline("coreference-resolution")
        out = coref(txt)
        # some implementations return 'resolved_text' key
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict) and "resolved_text" in out[0]:
            resolved = out[0]["resolved_text"]
            return resolved
        if isinstance(out, dict) and "resolved_text" in out:
            return out["resolved_text"]
    except Exception as e:
        st.warning(f"Coref resolution failed or not available: {e}")
    return txt

# attempt coref on a reduced chunk (full doc coref may be expensive)
try:
    if use_coref and COREF_AVAILABLE:
        # run coref on the whole text could be heavy; we run it if user asked
        text_for_search = try_coref_resolution(text_norm)
    else:
        text_for_search = text_norm
except Exception:
    text_for_search = text_norm

# --- quantulum3-based quantity extraction fallback regex ---
# Fallback numeric unit regex (robust)
num_pattern = r"(?P<number>[+\-]?\d{1,3}(?:[,\d{3}])*\.?\d*(?:[eE][+\-]?\d+)?|\d*\.?\d+(?:[eE][+\-]?\d+)?)"
units_pattern = r"(?:uV\/K|uV K(?:\^-1|⁻¹)?|mV\/K|mV K(?:\^-1|⁻¹)?|V\/K|microvolts per kelvin|microvolt per kelvin|µV\/K|μV\/K|uVK\^\{-1\})"
value_regex = re.compile(
    rf"(?P<prefix>[\s\(,\-\/:]*)?(?P<number_full>(?:±|\+\/\-)?\s*(?:{num_pattern})(?:\s*(?:to|–|-|—)\s*{num_pattern})?(?:\s*(?:±|\+\/\-)\s*{num_pattern})?)\s*(?P<unit>{units_pattern})",
    flags=re.IGNORECASE
)

# helper: extract quantities from a snippet using quantulum3 if available else regex fallback
def extract_quantities_from_text(snippet):
    results = []
    snippet_clean = snippet.replace("µ", "u").replace("μ", "u")
    if QUANTULUM_AVAILABLE:
        try:
            qts = quant_parser.parse(snippet_clean)
            for q in qts:
                if getattr(q, "unit", None) is None:
                    continue
                unit_name = str(q.unit)  # e.g., "microvolt"
                # look for volt-per-kelvin dimensions (heuristic)
                unit_str = q.surface  # surface text
                results.append({
                    "value": float(q.value),
                    "unit": unit_str,
                    "raw": q.surface,
                    "span": (q.span[0], q.span[1])
                })
            if results:
                return results
        except Exception:
            pass

    # fallback: regex finditer
    for m in value_regex.finditer(snippet_clean):
        raw = snippet[m.start():m.end()]
        # parse numbers and ranges
        # extract first number as representative, but also capture range if present
        number_match = re.search(r"[+\-]?\d[\d,\.eE+\-]*", m.group("number_full"))
        if number_match:
            ntext = number_match.group(0).replace(",", "")
            try:
                val = float(ntext)
            except Exception:
                val = None
        else:
            val = None
        results.append({
            "value": val,
            "unit": m.group("unit"),
            "raw": raw,
            "span": (m.start(), m.end())
        })
    return results

# --- material extraction helpers ---
# 1) try ChemDataExtractor
def chemdata_extract_materials(snippet):
    if not CHEMDATA_AVAILABLE:
        return []
    try:
        doc = CDE_Document(snippet)
        mats = []
        # CDE has .cems or .chemicals? We'll look for chemical entities heuristically
        # For safety we parse paragraphs and look for chemical names
        for p in doc.paragraphs:
            textp = p.text
            # gather rough tokens with capitalization and formula-like strings
            formula_like = re.findall(r"\b[A-Z][a-z]?[0-9\._\-\{\}^\$]*[A-Za-z0-9\}\)]*", textp)
            for f in formula_like:
                if len(f) > 1:
                    mats.append(f)
        return list(dict.fromkeys(mats))
    except Exception:
        return []

# 2) formula regex + pymatgen validation
formula_regex = re.compile(r"\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+|[A-Z][a-z]?\d+)\b")
def find_formula_candidates(snippet):
    candidates = []
    for m in formula_regex.finditer(snippet):
        token = m.group(0)
        candidates.append(token)
    # validate with pymatgen if available
    validated = []
    for c in candidates:
        if PYMATGEN_AVAILABLE:
            try:
                comp = Composition(c)
                if len(comp.elements) > 0:
                    validated.append(c)
                    continue
            except Exception:
                pass
        else:
            # accept if token looks like formula (has digits or multiple element caps)
            if re.search(r"[A-Z][a-z]?\d", c) or sum(1 for ch in c if ch.isupper()) > 1:
                validated.append(c)
    return list(dict.fromkeys(validated))

# 3) noun phrase heuristics using spaCy
def noun_phrase_candidates(snippet, max_candidates=5):
    if SPACY_AVAILABLE and nlp is not None:
        doc = nlp(snippet)
        nps = [chunk.text.strip() for chunk in doc.noun_chunks]
        # prefer longer multiword noun phrases or those containing 'alloy', 'compound', etc.
        good = [p for p in nps if len(p) > 2][:max_candidates]
        return list(dict.fromkeys(good))
    else:
        # crude heuristics: sequences of words with lowercase letters and hyphens
        tokens = re.findall(r"([A-Za-z][A-Za-z\-\s]{1,60})", snippet)
        candidates = [t.strip() for t in tokens if len(t.strip().split()) <= 6 and len(t.strip()) > 3]
        return candidates[:max_candidates]

# top-level material extraction around a numeric index (bidirectional)
def extract_material_around(value_abs_start, value_abs_end, full_text, max_window_chars=400):
    # look in a window around the numeric span (both before and after)
    start = max(0, value_abs_start - max_window_chars)
    end = min(len(full_text), value_abs_end + max_window_chars)
    window = full_text[start:end]
    # First: chemdataextractor
    mats = chemdata_extract_materials(window)
    if mats:
        return mats[0], "chemdataextractor"
    # Second: formula-like tokens validated
    formulas = find_formula_candidates(window)
    if formulas:
        return formulas[0], "formula_candidate"
    # Third: noun phrases
    nps = noun_phrase_candidates(window)
    if nps:
        return nps[0], "nounphrase"
    # fallback: try preceding sentence or next sentence
    segs = sentence_segments(full_text)
    # find containing sentence by char offset
    cum = 0
    for s in segs:
        s_text = s.text if hasattr(s, "text") else str(s)
        s_len = len(s_text) + 1
        if start <= cum + s_len and cum <= end:
            # take this sentence and neighbors
            idx = segs.index(s)
            neighbors = []
            try:
                neighbors = [segs[idx-1].text, s_text, segs[idx+1].text]
            except Exception:
                neighbors = [s_text]
            joined = " ".join([str(x) for x in neighbors if x])
            # try formula parse again
            formulas2 = find_formula_candidates(joined)
            if formulas2:
                return formulas2[0], "formula_sentence"
            nps2 = noun_phrase_candidates(joined)
            if nps2:
                return nps2[0], "nounphrase_sentence"
        cum += s_len
    return "", "none_found"

# --- Table extraction (Camelot/Tabula) ---
def extract_tables_text_from_pdf_file(pdf_path):
    table_texts = []
    # Camelot (works for lattice/stream tables)
    if CAMELOT_AVAILABLE:
        try:
            tables = camelot.read_pdf(pdf_path, pages='all')
            for t in tables:
                df = t.df
                # join rows into text
                table_texts.append("\n".join(["\t".join(list(r)) for r in df.values]))
        except Exception:
            pass
    # tabula (Java-based) fallback
    if TABULA_AVAILABLE:
        try:
            dfs = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            for df in dfs:
                if isinstance(df, pd.DataFrame):
                    table_texts.append(df.to_csv(sep="\t", index=False))
        except Exception:
            pass
    return table_texts

# If user asked to search tables, attempt to extract them and append to search corpus
tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
tmpf.write(uploaded_file.getbuffer())
tmpf.flush()
tmpf.close()
pdf_path_for_tables = tmpf.name

try:
    if search_tables:
        table_texts = extract_tables_text_from_pdf_file(pdf_path_for_tables)
        if table_texts:
            st.info(f"Found {len(table_texts)} tables. Searching them for Seebeck entries.")
            # append table text to main search corpus (tagged so we can show origin)
            text_for_search += "\n\n" + "\n\n".join(table_texts)
except Exception as e:
    st.warning(f"Table extraction failed or not available: {e}")

# --- Find occurrences of keywords at sentence granularity ---
# Build list of sentence objects
sentences = sentence_segments(text_for_search)

# prepare list of matches
entries = []

# iterate sentences (and optionally neighbors) to find keywords
for i, sent in enumerate(sentences):
    s_text = sent.text if hasattr(sent, "text") else str(sent)
    if keyword_re.search(s_text):
        # create search context: sentence plus optional neighbors
        context_text = s_text
        if expand_sentences:
            # include previous and next sentences to capture "material before/after"
            prev_s = sentences[i-1].text if i-1 >= 0 and hasattr(sentences[i-1], "text") else (sentences[i-1] if i-1 >= 0 else "")
            next_s = sentences[i+1].text if i+1 < len(sentences) and hasattr(sentences[i+1], "text") else (sentences[i+1] if i+1 < len(sentences) else "")
            context_text = " ".join([str(x) for x in [prev_s, s_text, next_s] if x])
        # Now extract quantities within this context using quantulum or fallback
        quant_matches = extract_quantities_from_text(context_text)
        # If none found in context, expand to larger window around this sentence (character-based)
        if not quant_matches:
            # find char index of this sentence in the original normalized text
            try:
                idx = text_for_search.index(s_text)
                window_start = max(0, idx - 600)
                window_end = min(len(text_for_search), idx + len(s_text) + 600)
                larger_snip = text_for_search[window_start:window_end]
                quant_matches = extract_quantities_from_text(larger_snip)
            except Exception:
                quant_matches = []
        # If still none and tables were added, attempt searching entire doc (may be noisy)
        if not quant_matches:
            quant_matches = extract_quantities_from_text(text_for_search)
        # For each found quantity, attempt to associate material
        if quant_matches:
            for qm in quant_matches:
                # Compute approximate absolute positions relative to full text_for_search (best-effort)
                # We'll search for the raw within the larger text to get an absolute index
                raw = qm.get("raw", "")
                try:
                    abs_start = text_for_search.index(raw)
                    abs_end = abs_start + len(raw)
                except Exception:
                    # fallback: approximate using sentence location
                    try:
                        s_idx = text_for_search.index(s_text)
                        abs_start = s_idx + (qm.get("span", (0,0))[0] if qm.get("span") else 0)
                        abs_end = abs_start + (qm.get("span", (0,0))[1] - qm.get("span", (0,0))[0] if qm.get("span") else len(raw))
                    except Exception:
                        abs_start = None
                        abs_end = None

                # extract material near this value (bidirectional)
                if abs_start is not None and abs_end is not None:
                    mat, mat_src = extract_material_around(abs_start, abs_end, text_for_search, max_window_chars=500)
                else:
                    mat, mat_src = "", "none_found"

                # parse signs/ranges from raw (±, to, -)
                raw_text = qm.get("raw", "")
                val_primary = qm.get("value", None)
                value_min, value_max, value_err = None, None, None
                # handle common patterns
                # ± pattern
                m_pm = re.search(r"([+\-]?\d[\d\.,eE\+\-]*)\s*[±]\s*([+\-]?\d[\d\.,eE\+\-]*)", raw_text)
                m_range = re.search(r"([+\-]?\d[\d\.,eE\+\-]*)\s*(?:to|–|-|—)\s*([+\-]?\d[\d\.,eE\+\-]*)", raw_text)
                if m_pm:
                    try:
                        v = float(m_pm.group(1).replace(",", ""))
                        err = float(m_pm.group(2).replace(",", ""))
                        val_primary = v
                        value_err = err
                    except Exception:
                        pass
                elif m_range:
                    try:
                        v1 = float(m_range.group(1).replace(",", ""))
                        v2 = float(m_range.group(2).replace(",", ""))
                        value_min = min(v1, v2)
                        value_max = max(v1, v2)
                        val_primary = (value_min + value_max) / 2.0
                    except Exception:
                        pass

                unit = qm.get("unit", None)
                # normalize unit simple mapping
                def normalize_unit(u):
                    if not u:
                        return None
                    u2 = str(u).lower()
                    if "uv" in u2 or "microvolt" in u2:
                        return "uV/K"
                    if "mv" in u2:
                        return "mV/K"
                    if "v" in u2 and "/k" in u2:
                        return "V/K"
                    return u

                unit_norm = normalize_unit(unit)

                # convert to microvolt per kelvin when possible
                def to_uV_per_K(val, unitstr):
                    if val is None:
                        return None
                    if unitstr is None:
                        return val
                    u = unitstr.lower()
                    try:
                        if "mv" in u:
                            return float(val) * 1000.0
                        if "uv" in u or "microvolt" in u:
                            return float(val)
                        if "v" in u and "/k" in u:
                            return float(val) * 1e6
                    except Exception:
                        return val
                    return val

                val_uV = to_uV_per_K(val_primary, unit_norm) if val_primary is not None else None
                val_min_uV = to_uV_per_K(value_min, unit_norm) if value_min is not None else None
                val_max_uV = to_uV_per_K(value_max, unit_norm) if value_max is not None else None

                entries.append({
                    "sentence_index": i,
                    "sentence_text": s_text,
                    "value_raw": raw_text,
                    "value_numeric": val_primary,
                    "value_min": value_min,
                    "value_max": value_max,
                    "value_err": value_err,
                    "unit": unit,
                    "unit_norm": unit_norm,
                    "value_uV_per_K": val_uV,
                    "value_min_uV_per_K": val_min_uV,
                    "value_max_uV_per_K": val_max_uV,
                    "material": mat,
                    "material_source": mat_src,
                    "abs_span": (abs_start, abs_end) if abs_start is not None else (None, None),
                })
        else:
            # no quantities found in this sentence block: create placeholder entry (optionally)
            entries.append({
                "sentence_index": i,
                "sentence_text": s_text,
                "value_raw": None,
                "value_numeric": None,
                "value_min": None,
                "value_max": None,
                "value_err": None,
                "unit": None,
                "unit_norm": None,
                "value_uV_per_K": None,
                "value_min_uV_per_K": None,
                "value_max_uV_per_K": None,
                "material": None,
                "material_source": "no_value_found",
                "abs_span": (None, None),
            })

# Build dataframe
df = pd.DataFrame(entries)

# Outlier detection (z-score) on the µV/K values
vals = df["value_uV_per_K"].dropna().astype(float)
if len(vals) >= 2:
    mean = vals.mean()
    std = vals.std(ddof=0) if vals.std(ddof=0) > 0 else 1.0
    df["zscore"] = df["value_uV_per_K"].apply(lambda v: (v - mean)/std if pd.notna(v) else None)
    df["is_outlier"] = df["zscore"].apply(lambda z: abs(z) > outlier_zscore_thresh if z is not None else False)
else:
    df["zscore"] = None
    df["is_outlier"] = False

# Show results in Streamlit
st.subheader("Extracted entries (one per found quantity / sentence)")
display_cols = [
    "sentence_index", "material", "material_source", "value_raw", "value_numeric", "value_min", "value_max",
    "unit", "unit_norm", "value_uV_per_K", "zscore", "is_outlier"
]
if df.shape[0] == 0:
    st.write("No entries extracted.")
else:
    st.dataframe(df[display_cols].fillna(""))

# Save to SQLite .db (in-memory then downloadable)
conn = sqlite3.connect(":memory:")
df.to_sql("seebeck_entries", conn, index=False, if_exists="replace")
# dump DB to bytes for download
with BytesIO() as b_io:
    for line in conn.iterdump():
        b_io.write(f"{line}\n".encode("utf-8"))
    sql_text = b_io.getvalue()
conn.close()
# Create a real .db file to download
tmp_db_path = "seebeck_data_output.db"
if os.path.exists(tmp_db_path):
    os.remove(tmp_db_path)
tmp_conn = sqlite3.connect(tmp_db_path)
df.to_sql("seebeck_entries", tmp_conn, index=False, if_exists="replace")
tmp_conn.commit()
tmp_conn.close()
with open(tmp_db_path, "rb") as f:
    db_data = f.read()
st.download_button("Download SQLite .db", db_data, file_name="seebeck_data.db", mime="application/octet-stream")

# CSV download
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv_bytes, file_name="seebeck_data.csv", mime="text/csv")

# Summary
st.subheader("Summary statistics")
if len(vals) > 0:
    st.write(f"Count (numeric): {len(vals)}")
    st.write(f"Mean (µV/K): {mean:.3g}")
    st.write(f"Median (µV/K): {vals.median():.3g}")
    st.write(f"Std (µV/K): {std:.3g}")
    st.write("Outliers flagged:", int(df["is_outlier"].sum()))
else:
    st.write("No numeric Seebeck values parsed.")

# Clean up tmp pdf and db if created
try:
    os.unlink(pdf_path_for_tables)
except Exception:
    pass
try:
    os.remove(tmp_db_path)
except Exception:
    pass

# Notes and instructions
st.markdown("**Notes / behavior**")
st.markdown("""
- The app uses sentence-level context and optionally neighboring sentences to find quantities and candidate materials.
- If `quantulum3` is installed it will be used for unit-aware parsing; otherwise a regex fallback is used.
- ChemDataExtractor and pymatgen provide improved chemical/material recognition when available.
- Table extraction requires Camelot or Tabula to be installed and may require additional system packages.
- Coref resolution via Hugging Face transformer pipeline is optional and may take long / require downloading a large model.
- This design is hybrid: rules + domain NER + unit parser; it's robust for many cases but still imperfect for complex layouts (figures/embedded images).  
""")

st.success("Extraction completed. Inspect results and download CSV/DB for review.")
