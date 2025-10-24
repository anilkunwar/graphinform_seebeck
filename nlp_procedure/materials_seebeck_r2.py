import streamlit as st
import re
import sqlite3
import pandas as pd
from io import BytesIO
import pubchempy as pcp

# PDF text extraction
try:
    import pdfplumber
    PDF_PLUMBER_AVAILABLE = True
except ImportError:
    PDF_PLUMBER_AVAILABLE = False
    st.warning("⚠️ pdfplumber not available. Install it with: pip install pdfplumber")

# Quantity parser (with Python 3.13-safe fix)
try:
    from quantulum3 import parser as qparser
    QUANTULUM_AVAILABLE = True
except ImportError:
    QUANTULUM_AVAILABLE = False
    st.warning("⚠️ quantulum3 not available. Install it with: pip install quantulum3")

# ──────────────────────────────────────────────────────────────
# Streamlit App UI
# ──────────────────────────────────────────────────────────────
st.title("📘 PDF → SQLite/CSV: Seebeck Coefficient Extractor")
st.markdown("""
This tool extracts **Seebeck coefficients** and their associated **materials** from scientific PDF documents.
It also retrieves **chemical metadata** via PubChem.
""")

uploaded_file = st.file_uploader("📂 Upload a PDF file", type="pdf")

# ──────────────────────────────────────────────────────────────
# Helper: Extract text
# ──────────────────────────────────────────────────────────────
def extract_pdf_text(uploaded_file):
    if not PDF_PLUMBER_AVAILABLE:
        st.error("pdfplumber is required but not installed.")
        return ""
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# ──────────────────────────────────────────────────────────────
# Helper: Material metadata from PubChem
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_pubchem_metadata(material_name):
    try:
        compounds = pcp.get_compounds(material_name, "name")
        if not compounds:
            return None
        comp = compounds[0]
        return {
            "PubChem_ID": comp.cid,
            "Molecular_Formula": comp.molecular_formula,
            "Molecular_Weight": comp.molecular_weight,
            "IUPAC_Name": comp.iupac_name,
        }
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────
# Core Data Extraction Function
# ──────────────────────────────────────────────────────────────
def extract_data(text):
    seebeck_pattern = r"seebeck coefficient|thermoelectric power|S coefficient"
    value_pattern = r"([+-]?\d+(?:\.\d+)?)\s*(µV/K|μV/K|uV/K|mV/K|V/K|μV\s*K⁻¹|uV\s*K⁻¹|mV\s*K⁻¹|V\s*K⁻¹)"
    material_pattern = r"\b([A-Z][a-z]?\d*[A-Za-z\d]*(?:\s*[-–]\s*[A-Z][a-z]?\d*[A-Za-z\d]*)*)\b"

    results = []

    for match in re.finditer(seebeck_pattern, text, re.IGNORECASE):
        start_pos = match.start()
        context_start = max(0, start_pos - 250)
        context_end = min(len(text), start_pos + 400)
        context = text[context_start:context_end]

        # Look for numeric Seebeck value
        value_match = re.search(value_pattern, context)
        value = None
        unit = None
        if value_match:
            try:
                value = float(value_match.group(1))
            except ValueError:
                value = None
            unit = value_match.group(2)

        # Material extraction before Seebeck mention
        material_context = text[max(0, start_pos - 300):start_pos]
        material_matches = re.findall(material_pattern, material_context)
        material = material_matches[-1] if material_matches else None

        # Quantulum3 extraction (Python 3.13–safe)
        quantities = []
        if QUANTULUM_AVAILABLE:
            try:
                quantities = qparser.parse(context, classifier_path=None)
            except Exception:
                quantities = []

        # PubChem enrichment
        pubchem_data = get_pubchem_metadata(material) if material else None

        results.append({
            "Material": material,
            "Seebeck_Value": value,
            "Unit": unit,
            "Context": context.replace("\n", " "),
            "Quantities": "; ".join([str(q) for q in quantities]) if quantities else None,
            "PubChem_ID": pubchem_data["PubChem_ID"] if pubchem_data else None,
            "Molecular_Formula": pubchem_data["Molecular_Formula"] if pubchem_data else None,
            "Molecular_Weight": pubchem_data["Molecular_Weight"] if pubchem_data else None,
            "IUPAC_Name": pubchem_data["IUPAC_Name"] if pubchem_data else None,
        })

    return pd.DataFrame(results)

# ──────────────────────────────────────────────────────────────
# Main Workflow
# ──────────────────────────────────────────────────────────────
if uploaded_file is not None:
    text = extract_pdf_text(uploaded_file)
    if not text.strip():
        st.error("No readable text found in the PDF.")
        st.stop()

    st.subheader("🧾 Extracted Text Preview")
    st.text_area("Preview", text[:1500], height=300)

    st.subheader("🔍 Extracting Seebeck Coefficients...")
    df = extract_data(text)

    if not df.empty:
        st.success(f"✅ Found {len(df)} Seebeck coefficient entries.")
        st.dataframe(df)

        # ─── Database + CSV Export
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv, file_name="seebeck_data.csv", mime="text/csv")

        sqlite_bytes = BytesIO()
        conn = sqlite3.connect(":memory:")
        df.to_sql("seebeck_data", conn, index=False, if_exists="replace")
        with sqlite3.connect(sqlite_bytes) as mem_conn:
            df.to_sql("seebeck_data", mem_conn, index=False, if_exists="replace")
        sqlite_bytes.seek(0)
        st.download_button("💾 Download SQLite Database", data=sqlite_bytes.getvalue(),
                           file_name="seebeck_data.db", mime="application/x-sqlite3")

        # ─── Summary
        st.subheader("📊 Summary")
        st.write(f"**Entries with numeric values:** {df['Seebeck_Value'].notna().sum()}")
        st.write(f"**Entries with materials identified:** {df['Material'].notna().sum()}")
        st.write(f"**Entries enriched with PubChem data:** {df['PubChem_ID'].notna().sum()}")

    else:
        st.warning("⚠️ No Seebeck coefficients detected in this PDF.")

else:
    st.info("Please upload a PDF file to begin.")
