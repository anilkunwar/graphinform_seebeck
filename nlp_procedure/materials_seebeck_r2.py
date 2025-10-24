import streamlit as st
import re
import sqlite3
import pandas as pd
from io import BytesIO

# PDF processing
try:
    import pdfplumber
    PDF_PLUMBER_AVAILABLE = True
except ImportError:
    PDF_PLUMBER_AVAILABLE = False
    st.warning("pdfplumber not available. Install with: pip install pdfplumber")

st.title("PDF to SQLite/CSV: Seebeck Coefficient Extractor")
st.markdown("Upload a PDF to extract Seebeck coefficients and corresponding materials")

uploaded_file = st.file_uploader("Choose PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    text = ""
    if PDF_PLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            text = ""
    else:
        st.error("pdfplumber is required but not installed")
        st.stop()
    
    if not text.strip():
        st.error("No text could be extracted from the PDF")
        st.stop()
    
    st.subheader("Extracted Text Preview")
    st.text_area("Preview", text[:1500], height=300)
    
    # Patterns for finding Seebeck coefficients and materials
    seebeck_pattern = r"seebeck coefficient|thermoelectric power|S coefficient"
    value_pattern = r"([+-]?\d+\.?\d*)\s*(µV/K|μV/K|uV/K|mV/K|V/K|μV\s*K⁻¹|uV\s*K⁻¹|mV\s*K⁻¹|V\s*K⁻¹)"
    material_pattern = r"\b([A-Z][a-z]?\d*[A-Za-z\d]*(?:\s*[-\–]\s*[A-Z][a-z]?\d*[A-Za-z\d]*)*)\b"
    
    # Find all Seebeck coefficient mentions
    seebeck_matches = []
    for match in re.finditer(seebeck_pattern, text, re.IGNORECASE):
        start_pos = match.start()
        # Get context around the mention
        context_start = max(0, start_pos - 200)
        context_end = min(len(text), start_pos + 400)
        context = text[context_start:context_end]
        
        # Look for value in the context
        value_match = re.search(value_pattern, context)
        value = None
        unit = None
        if value_match:
            value = float(value_match.group(1))
            unit = value_match.group(2)
        
        # Look for material name before the Seebeck mention
        material = None
        material_context = text[max(0, start_pos-300):start_pos]
        material_matches = re.findall(material_pattern, material_context)
        if material_matches:
            # Take the last material mentioned before Seebeck
            material = material_matches[-1] if material_matches else None
        
        seebeck_matches.append({
            'context': context.replace('\n', ' '),
            'value': value,
            'unit': unit,
            'material': material,
            'position': start_pos
        })
    
    # Create DataFrame
    df = pd.DataFrame(seebeck_matches)
    
    if not df.empty:
        st.subheader("Extracted Data")
        st.dataframe(df)
        
        # Create SQLite database
        conn = sqlite3.connect(':memory:')
        df.to_sql('seebeck_data', conn, index=False, if_exists='replace')
        
        # Save SQLite to bytes for download
        sqlite_bytes = BytesIO()
        with sqlite3.connect(sqlite_bytes) as mem_conn:
            df.to_sql('seebeck_data', mem_conn, index=False, if_exists='replace')
        sqlite_bytes.seek(0)
        
        # Download buttons
        st.download_button(
            label="Download SQLite Database",
            data=sqlite_bytes.getvalue(),
            file_name="seebeck_data.db",
            mime="application/x-sqlite3"
        )
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="seebeck_data.csv",
            mime="text/csv"
        )
        
        # Summary statistics
        st.subheader("Summary")
        st.write(f"Total Seebeck coefficient mentions found: {len(df)}")
        st.write(f"Entries with values: {df['value'].notna().sum()}")
        st.write(f"Entries with materials: {df['material'].notna().sum()}")
        
    else:
        st.info("No Seebeck coefficients found in the document")

else:
    st.info("Please upload a PDF file to begin")
