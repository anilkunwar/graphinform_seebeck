import streamlit as st
import re
import sqlite3
import pandas as pd
from io import BytesIO

# Optional NLP & parsing tools
import spacy
from quantulum3 import parser as qparser
from chemdataextractor import Document

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

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

st.title("Seebeck Coefficient Extractor v3")
st.markdown("📘 Upload a scientific PDF to extract Seebeck coefficients and related materials automatically.")

uploaded_file = st.file_uploader("📄 Choose a PDF file", type="pdf")

# --- Helper Functions ---
def extract_text_from_pdf(file):
    """Extract text using pdfplumber"""
    text = ""
    if not PDF_PLUMBER_AVAILABLE:
        return ""
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
    """Extract tables using Camelot"""
    if not CAMELOT_AVAILABLE:
        return pd.DataFrame()
    try:
        tables = camelot.read_pdf(file.name, pages="all")
        all_data = pd.concat([t.df for t in tables], ignore_index=True)
        return all_data
    except Exception:
        return pd.DataFrame()

def find_nearest_material(sentence_text, materials, value_pos):
    """Find the material closest to a Seebeck value in a sentence"""
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
    """Extract Seebeck coefficient data from text"""
    results = []
    doc = nlp(text)
    cdoc = Document(text)
    materials = [cem.text for cem in cdoc.cems]
    materials = list(set(materials))  # unique list

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if re.search(r"(seebeck|thermoelectric power|S[-\s]?value)", sent_text, re.IGNORECASE):
            # Use Quantulum3 for robust quantity parsing
            quantities = qparser.parse(sent_text)
            for q in quantities:
                if (
                    q.unit
                    and any(u in str(q.unit).lower() for u in ["volt", "µv", "μv"])
                    and "kelvin" in str(q.unit).lower()
                ):
                    value_pos = sent_text.lower().find(str(q.value))
                    nearest_material = find_nearest_material(sent_text, materials, value_pos)
                    results.append({
                        "Sentence": sent_text,
                        "Material": nearest_material if nearest_material else "Unknown",
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
                nearest_material = find_nearest_material(sent_text, materials, vm.start())
                results.append({
                    "Sentence": sent_text,
                    "Material": nearest_material if nearest_material else "Unknown",
                    "Seebeck_value": val,
                    "Unit": unit,
                    "Confidence": "Medium (Regex)"
                })
    return pd.DataFrame(results)

# --- Main logic ---
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    if not text.strip():
        st.error("No readable text found in the PDF.")
        st.stop()

    st.subheader("📄 Extracted Text Preview")
    st.text_area("Preview", text[:1500], height=300)

    with st.spinner("Extracting Seebeck data..."):
        df = extract_data(text)

    # Combine with table data if available
    tables_df = extract_tables_from_pdf(uploaded_file)
    if not tables_df.empty:
        st.info(f"Extracted {len(tables_df)} table rows (Camelot).")
        # Search for columns mentioning Seebeck
        seebeck_cols = [c for c in tables_df.columns if re.search("seebeck", c, re.IGNORECASE)]
        if seebeck_cols:
            st.write("Detected Seebeck columns:", seebeck_cols)

    if not df.empty:
        st.success(f"✅ Extracted {len(df)} Seebeck entries")
        st.dataframe(df)

        # Save to SQLite memory
        conn = sqlite3.connect(':memory:')
        df.to_sql('seebeck_data', conn, index=False, if_exists='replace')

        # Save SQLite for download
        sqlite_bytes = BytesIO()
        with sqlite3.connect(sqlite_bytes) as mem_conn:
            df.to_sql('seebeck_data', mem_conn, index=False, if_exists='replace')
        sqlite_bytes.seek(0)

        # Download buttons
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
        st.subheader("📊 Summary Statistics")
        st.write(f"Total mentions found: {len(df)}")
        st.write(f"Entries with material identified: {df['Material'].ne('Unknown').sum()}")
        st.write(f"Mean Seebeck value (approx.): {df['Seebeck_value'].mean():.2f} µV/K" if df['Seebeck_value'].notna().any() else "N/A")

    else:
        st.warning("No Seebeck coefficients detected.")
else:
    st.info("Please upload a PDF file to start.")
