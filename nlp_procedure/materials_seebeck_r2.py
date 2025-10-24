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
        uploaded_file.seek(0)  # Ensure file pointer is at the start
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
    # Skip non-chemical strings that might be captured by material_pattern
    if not material_name or len(material_name.split()) > 4:
        return None
    try:
        compounds = pcp.get_compounds(material_name, "name")
        if not compounds:
            # Try searching by formula instead
            compounds = pcp.get_compounds(material_name, "formula")
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
# Helper: Convert value to standard units (µV/K)
# ──────────────────────────────────────────────────────────────
def convert_to_standard_units(value, unit):
    """Convert Seebeck value to standard µV/K units"""
    if value is None or unit is None:
        return None
    
    unit = unit.lower().strip()
    value = float(value)
    
    conversion_factors = {
        'µv/k': 1.0,
        'μv/k': 1.0,
        'uv/k': 1.0,
        'mv/k': 1000.0,
        'v/k': 1000000.0,
        'µv k⁻¹': 1.0,
        'μv k⁻¹': 1.0,
        'uv k⁻¹': 1.0,
        'mv k⁻¹': 1000.0,
        'v k⁻¹': 1000000.0,
        'µv k^{-1}': 1.0,
        'μv k^{-1}': 1.0,
    }
    
    for unit_pattern, factor in conversion_factors.items():
        if unit_pattern in unit:
            return value * factor
    
    return value  # Return original if unit not recognized

# ──────────────────────────────────────────────────────────────
# Core Data Extraction Function
# ──────────────────────────────────────────────────────────────
def extract_data(text):
    seebeck_pattern = r"seebeck coefficient|thermoelectric power|S coefficient|Seebeck"
    
    # Improved value pattern to capture various formats
    value_pattern = r"([+-]?\d+(?:[\.,]\d+)?(?:[eE][+-]?\d+)?)\s*(?:±\s*\d+)?\s*(µV/K|μV/K|uV/K|mV/K|V/K|μV\s*K[⁻¹\-1]|uV\s*K[⁻¹\-1]|mV\s*K[⁻¹\-1]|V\s*K[⁻¹\-1]|μV\s*K\^{?[-\–]?1}?|uV\s*K\^{?[-\–]?1}?)"
    
    # Improved material pattern for chemical formulas and names
    material_pattern = r"\b(?:[A-Z][a-z]?\d*)+(?:[-–]\s*(?:[A-Z][a-z]?\d*)+)*\b"
    
    # Common words to exclude from material matches
    exclude_words = {'fig', 'table', 'reference', 'et al', 'al', 'and', 'the', 
                    'for', 'with', 'from', 'this', 'that', 'these', 'those',
                    'using', 'based', 'method', 'study', 'research', 'paper'}

    results = []

    for match in re.finditer(seebeck_pattern, text, re.IGNORECASE):
        start_pos = match.start()
        context_start = max(0, start_pos - 300)
        context_end = min(len(text), start_pos + 500)
        context = text[context_start:context_end]

        # Look for numeric Seebeck value
        value_match = re.search(value_pattern, context, re.IGNORECASE)
        value = None
        unit = None
        value_raw = None
        value_standard = None
        
        if value_match:
            value_raw = value_match.group(1).strip()
            unit = value_match.group(2).strip()
            
            # Clean and convert value
            try:
                # Replace comma with dot for European format
                clean_value = value_raw.replace(',', '.')
                # Handle scientific notation
                if 'e' in clean_value.lower():
                    value = float(clean_value)
                else:
                    # Remove any non-numeric characters except dots and minus
                    clean_value = re.sub(r'[^\d\.\-]', '', clean_value)
                    value = float(clean_value)
                
                # Convert to standard units
                value_standard = convert_to_standard_units(value, unit)
                
            except (ValueError, TypeError):
                value = None
                value_standard = None

        # Material extraction - look in wider context
        material = None
        material_context = text[max(0, start_pos - 500):start_pos + 200]
        material_matches = re.findall(material_pattern, material_context)
        
        # Filter and select the best material candidate
        valid_materials = []
        for m in material_matches:
            m_clean = m.strip()
            # Exclude common words and very short strings
            if (m_clean.lower() not in exclude_words and 
                len(m_clean) > 1 and 
                not m_clean.isdigit() and
                not m_clean.endswith(('.', ',', ';', ':'))):
                valid_materials.append(m_clean)
        
        # Prefer materials that look like chemical formulas
        chemical_like = [m for m in valid_materials if re.search(r'[A-Z][a-z]?\d*[A-Z]', m)]
        if chemical_like:
            material = chemical_like[-1]  # Take the last one (closest to mention)
        elif valid_materials:
            material = valid_materials[-1]

        # PubChem enrichment
        pubchem_data = get_pubchem_metadata(material) if material else None

        results.append({
            "Material": material,
            "Seebeck_Value_Raw": value_raw,
            "Seebeck_Value_Numeric": value,
            "Seebeck_Unit": unit,
            "Seebeck_Value_µV_K": value_standard,
            "Context": context.replace("\n", " ")[:500] + "...",  # Limit context length
            "PubChem_ID": pubchem_data.get("PubChem_ID") if pubchem_data else None,
            "Molecular_Formula": pubchem_data.get("Molecular_Formula") if pubchem_data else None,
            "Molecular_Weight": pubchem_data.get("Molecular_Weight") if pubchem_data else None,
            "IUPAC_Name": pubchem_data.get("IUPAC_Name") if pubchem_data else None,
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
    
    with st.spinner('Parsing text and performing enrichment...'):
        df = extract_data(text)

    if not df.empty:
        st.success(f"✅ Found {len(df)} Seebeck coefficient entries.")
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Entries", len(df))
        with col2:
            st.metric("With Values", df['Seebeck_Value_Numeric'].notna().sum())
        with col3:
            st.metric("With Materials", df['Material'].notna().sum())
        with col4:
            st.metric("PubChem Matches", df['PubChem_ID'].notna().sum())

        # Reorder columns for better display
        display_cols = [
            "Material", "Seebeck_Value_Numeric", "Seebeck_Unit", 
            "Seebeck_Value_µV_K", "Molecular_Formula", "PubChem_ID", 
            "Context"
        ]
        # Ensure all required columns exist before displaying
        available_cols = [col for col in display_cols if col in df.columns]
        df_display = df[available_cols]
        
        st.dataframe(df_display)

        # ─── Database + CSV Export
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV", 
            data=csv, 
            file_name="seebeck_data.csv", 
            mime="text/csv"
        )

        # Export to SQLite
        sqlite_bytes = BytesIO()
        conn = sqlite3.connect(sqlite_bytes, check_same_thread=False)
        df.to_sql("seebeck_data", conn, index=False, if_exists="replace")
        conn.commit()
        conn.close()
        
        st.download_button(
            "💾 Download SQLite Database", 
            data=sqlite_bytes.getvalue(),
            file_name="seebeck_data.db", 
            mime="application/x-sqlite3"
        )

    else:
        st.warning("⚠️ No Seebeck coefficients detected in this PDF.")

else:
    st.info("Please upload a PDF file to begin.")
