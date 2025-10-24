import streamlit as st
import re
import sqlite3
import pandas as pd
from io import BytesIO

import spacy
from quantulum3 import parser as qparser
import pubchempy as pcp

# Optional PDF tools
try:
    import pdfplumber
    PDF_PLUMBER_AVAILABLE = True
except ImportError:
    PDF_PLUMBER_AVAILABLE = False
    st.warning("pdfplumber not available. Install with: pip install pdfplumber")

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# --- Streamlit UI ---
st.title("Seebeck Coefficient Extractor v4 (PubChem-Enhanced)")
st.markdown("📘 Upload a PDF to extract Seebeck coefficients and validate materials using PubChem.")

uploaded_file = st.file_uploader("📄 Choose a PDF file", type="pdf")


# ==========================
# 🔧 Helper Functions
# ==========================

def extract_text_from_pdf(file):
    """Extract text using pdfplumber."""
    text = ""
    if not PDF_PLUMBER_AVAILABLE:
        return text
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text


def extract_tables_from_pdf(file):
    """Extract tabular data (optional) using Camelot."""
    if not CAMELOT_AVAILABLE:
        return pd.DataFrame()
    try:
        tables = camelot.read_pdf(file.name, pages="all")
        all_data = pd.concat([t.df for t in tables], ignore_index=True)
        return all_data
    except Exception:
        return pd.DataFrame()


def detect_material_candidates(text):
    """Detect possible material names or chemical formulas using regex."""
    # Simple patterns: chemical formulas (like Bi2Te3, SnSe, CuInSe2) and capitalized material names
    formula_pattern = r"\b([A-Z][a-z]?\d*[A-Za-z\d]*(?:[-–][A-Z][a-z]?\d*[A-Za-z\d]*)*)\b"
    name_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"

    formulas = re.findall(formula_pattern, text)
    names = re.findall(name_pattern, text)
    candidates = list(set(formulas + names))
    return [c for c in candidates if len(c) > 1 and len(c) < 40]


def validate_with_pubchem(candidate):
    """Validate a material name/formula using PubChem API."""
    try:
        results = pcp.get_compounds(candidate, 'name')
        if results:
            comp = results[0]
            return {
                "Material": candidate,
                "PubChem_ID": comp.cid,
                "Verified_Name": comp.iupac_name or comp.synonyms[0] if comp.synonyms else None,
            }
    except Exception:
        pass
    return None


def find_nearest_material(sentence_text, materials, value_pos):
    """Find the material closest to the Seebeck value within a sentence."""
    nearest = None
    min_dist = float("inf")
    for mat in materials:
        idx = sentence_text.lower().find(mat.lower())
        if idx != -1:
            dist = abs(idx - value_pos)
            if dist < min_dist:
                min_dist = dist
                nearest = mat
    return nearest


def extract_data(text):
    """Extract Seebeck coefficient data from text."""
    results = []
    doc = nlp(text)

    # --- Step 1: Detect and validate materials ---
    st.info("🔍 Validating material names with PubChem...")
    candidates = detect_material_candidates(text)
    validated_materials = []
    for c in candidates[:50]:  # limit to avoid API overload
        val = validate_with_pubchem(c)
        if val:
            validated_materials.append(val["Material"])
    validated_materials = list(set(validated_materials))

    # --- Step 2: Find Seebeck values per sentence ---
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if re.search(r"(seebeck|thermoelectric power|S[-\s]?value)", sent_text, re.IGNORECASE):
            # Use Quantulum3 for robust unit parsing
            quantities = qparser.parse(sent_text)
            for q in quantities:
                if (
                    q.unit
                    and any(u in str(q.unit).lower() for u in ["volt", "µv", "μv"])
                    and "kelvin" in str(q.unit).lower()
                ):
                    value_pos = sent_text.lower().find(str(q.value))
                    nearest_mat = find_nearest_material(sent_text, validated_materials, value_pos)
                    results.append({
                        "Sentence": sent_text,
                        "Material": nearest_mat if nearest_mat else "Unknown",
                        "Seebeck_value": q.value,
                        "Unit": str(q.unit),
                        "Confidence": "High (Quantulum)"
                    })

            # Fallback regex detection
            value_regex = re.compile(
                r"([+-]?\d*\.?\d+(?:\s*(?:±|\+\/\-)\s*\d+\.?\d*)?)\s*(µV/K|μV/K|uV/K|mV/K|V/K|μV·K⁻¹|µV·K⁻¹|microvolt(?:s)? per kelvin)",
                re.IGNORECASE
            )
            for vm in value_regex.finditer(sent_text):
                val, unit = vm.groups()
                try:
                    val = float(re.findall(r"[-+]?\d*\.?\d+", val)[0])
                except Exception:
                    val = None
                nearest_mat = find_nearest_material(sent_text, validated_materials, vm.start())
                results.append({
                    "Sentence": sent_text,
                    "Material": nearest_mat if nearest_mat else "Unknown",
                    "Seebeck_value": val,
                    "Unit": unit,
                    "Confidence": "Medium (Regex)"
                })
    return pd.DataFrame(results)


# ==========================
# 🚀 Main Logic
# ==========================

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    if not text.strip():
        st.error("No readable text found in the PDF.")
        st.stop()

    st.subheader("📄 Extracted Text Preview")
    st.text_area("Preview", text[:1500], height=300)

    with st.spinner("Extracting and validating data..."):
        df = extract_data(text)

    # Optional: extract tables
    tables_df = extract_tables_from_pdf(uploaded_file)
    if not tables_df.empty:
        st.info(f"📊 Extracted {len(tables_df)} table rows using Camelot.")

    if not df.empty:
        st.success(f"✅ Extracted {len(df)} Seebeck entries")
        st.dataframe(df)

        # Save SQLite database
        conn = sqlite3.connect(':memory:')
        df.to_sql('seebeck_data', conn, index=False, if_exists='replace')

        sqlite_bytes = BytesIO()
        with sqlite3.connect(sqlite_bytes) as mem_conn:
            df.to_sql('seebeck_data', mem_conn, index=False, if_exists='replace')
        sqlite_bytes.seek(0)

        st.download_button(
            "⬇️ Download SQLite Database",
            data=sqlite_bytes.getvalue(),
            file_name="seebeck_data.db",
            mime="application/x-sqlite3"
        )

        csv_data = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV",
            data=csv_data,
            file_name="seebeck_data.csv",
            mime="text/csv"
        )

        # Summary
        st.subheader("📈 Summary Statistics")
        st.write(f"Total Seebeck mentions: {len(df)}")
        st.write(f"Entries with materials identified: {df['Material'].ne('Unknown').sum()}")
        if df['Seebeck_value'].notna().any():
            st.write(f"Average Seebeck coefficient: {df['Seebeck_value'].mean():.2f} µV/K")
    else:
        st.warning("No Seebeck data found.")

else:
    st.info("Please upload a PDF file to start.")
