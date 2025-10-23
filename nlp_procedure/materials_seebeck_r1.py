import streamlit as st
import pdfplumber
import sqlite3
import pandas as pd
import re
import io

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def create_sqlite_db(text, db_filename):
    """Create SQLite database from extracted text"""
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdf_content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_number INTEGER,
            content TEXT
        )
    ''')
    
    # Split text by pages and insert
    pages = text.split('\n\n')  # Simple page separation
    for i, page_content in enumerate(pages):
        if page_content.strip():
            cursor.execute(
                "INSERT INTO pdf_content (page_number, content) VALUES (?, ?)",
                (i + 1, page_content.strip())
            )
    
    conn.commit()
    return conn

def extract_seebeck_data(text):
    """Extract Seebeck coefficients and material names using NER patterns"""
    # Patterns for Seebeck coefficient (various formats)
    seebeck_patterns = [
        r'Seebeck[\s\w]*coefficient[\s\w:]*([-+]?\d*\.?\d+)\s*([µμ]?V/K|V/K|mV/K)',
        r'([-+]?\d*\.?\d+)\s*([µμ]?V/K|V/K|mV/K)[\s\w]*Seebeck',
        r'S[\s]*=[\s]*([-+]?\d*\.?\d+)\s*([µμ]?V/K|V/K|mV/K)',
        r'([-+]?\d*\.?\d+)[\s~]*([µμ]?V/K|V/K|mV/K)'
    ]
    
    # Material name patterns (look for words before Seebeck mentions)
    material_pattern = r'([A-Z][a-z]*(?:\s*[A-Z]?[a-z]*)*(?:\s*[A-Z][a-z]*)*)[\s\w,]*Seebeck'
    
    results = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        
        # Look for Seebeck coefficients
        for pattern in seebeck_patterns:
            matches = re.finditer(pattern, line_clean, re.IGNORECASE)
            for match in matches:
                value = match.group(1)
                unit = match.group(2)
                
                # Look for material name in current or previous lines
                material = find_material_name(lines, i, line_clean)
                
                if material:
                    results.append({
                        'material': material,
                        'seebeck_coefficient': value,
                        'unit': unit,
                        'context': line_clean[:100] + '...' if len(line_clean) > 100 else line_clean
                    })
    
    return results

def find_material_name(lines, current_idx, current_line):
    """Find material name in current or preceding lines"""
    # Material patterns
    material_keywords = [
        r'([A-Z][a-z]*(?:\s*[A-Z]?[a-z]*)*(?:\s*[A-Z][a-z]*)*)\s*(?:compound|material|sample|film|alloy)',
        r'(?:material|sample)[\s]*:[\s]*([A-Z][a-z]*(?:\s*[A-Z]?[a-z]*)*)',
        r'([A-Z][a-z]*(?:\s*[A-Z]?[a-z]*)*)[\s]*(?:\d|\(|\))'  # Chemical formulas
    ]
    
    # Check current line first
    for pattern in material_keywords:
        match = re.search(pattern, current_line)
        if match:
            material = match.group(1).strip()
            if len(material) > 1 and len(material) < 50:  # Reasonable length for material name
                return material
    
    # Check previous lines (up to 3 lines back)
    for i in range(max(0, current_idx - 3), current_idx):
        for pattern in material_keywords:
            match = re.search(pattern, lines[i])
            if match:
                material = match.group(1).strip()
                if len(material) > 1 and len(material) < 50:
                    return material
    
    return None

def main():
    st.title("PDF to SQLite Converter with Seebeck Coefficient Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
    
    if uploaded_file is not None:
        st.success("PDF file uploaded successfully!")
        
        # Extract text from PDF
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        
        # Create SQLite database
        db_filename = "pdf_content.db"
        with st.spinner("Creating SQLite database..."):
            conn = create_sqlite_db(text, db_filename)
        
        # Display database info
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM pdf_content")
        row_count = cursor.fetchone()[0]
        
        st.subheader("Database Information")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Rows in database: {row_count}")
        with col2:
            # Download database
            with open(db_filename, "rb") as f:
                st.download_button(
                    label="Download SQLite Database",
                    data=f,
                    file_name="pdf_content.db",
                    mime="application/octet-stream"
                )
        
        # Display sample content
        st.subheader("Sample Content from Database")
        df_sample = pd.read_sql_query("SELECT * FROM pdf_content LIMIT 5", conn)
        st.dataframe(df_sample)
        
        # Perform NER analysis for Seebeck coefficients
        st.subheader("Seebeck Coefficient Analysis")
        with st.spinner("Analyzing Seebeck coefficients and materials..."):
            seebeck_data = extract_seebeck_data(text)
        
        if seebeck_data:
            st.success(f"Found {len(seebeck_data)} Seebeck coefficient entries!")
            
            # Create results dataframe
            df_results = pd.DataFrame(seebeck_data)
            
            # Display results
            st.dataframe(df_results)
            
            # Create CSV for download
            csv_buffer = io.StringIO()
            df_results.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="Download Results as CSV",
                data=csv_buffer.getvalue(),
                file_name="seebeck_analysis_results.csv",
                mime="text/csv"
            )
            
            # Summary statistics
            st.subheader("Summary Statistics")
            try:
                df_results['seebeck_value'] = pd.to_numeric(df_results['seebeck_coefficient'])
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Seebeck Coefficient", 
                             f"{df_results['seebeck_value'].mean():.2f} μV/K")
                with col2:
                    st.metric("Max Seebeck Coefficient", 
                             f"{df_results['seebeck_value'].max():.2f} μV/K")
                with col3:
                    st.metric("Min Seebeck Coefficient", 
                             f"{df_results['seebeck_value'].min():.2f} μV/K")
            except:
                st.warning("Could not calculate statistics - non-numeric values found")
        else:
            st.warning("No Seebeck coefficients found in the PDF. The patterns might not match your document format.")
            
            # Show sample of text for debugging
            with st.expander("Show extracted text sample for debugging"):
                st.text_area("First 2000 characters of extracted text:", 
                           text[:2000], height=300)
        
        conn.close()

if __name__ == "__main__":
    main()
