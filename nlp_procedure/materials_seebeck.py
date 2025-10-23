import streamlit as st
import re
import sqlite3
import pandas as pd
from io import StringIO
from PyPDF2 import PdfReader

st.title("📘 PDF → SQLite + NER Extraction (Seebeck Coefficient)")

# Step 1: Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Step 2: Extract text from PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"

    st.subheader("📄 Extracted Text Preview")
    st.text_area("Text content", text[:2000], height=200)

    # Step 3: NER pattern detection for Seebeck coefficient and materials
    # Common patterns: 250 µV/K, -120 μV/K, 1.2 mV/K etc.
    seebeck_pattern = r"([A-Za-z0-9\-\s]+)\s*[:\-]?\s*([-+]?\d*\.?\d+)\s*(µV\/K|uV\/K|μV\/K|mV\/K|V\/K)"
    matches = re.findall(seebeck_pattern, text)

    if matches:
        # Create DataFrame
        df = pd.DataFrame(matches, columns=["Material_Name", "Seebeck_Value", "Unit"])

        # Clean up Material names (take last word or clean symbols)
        df["Material_Name"] = df["Material_Name"].str.strip().str.replace(r"[^A-Za-z0-9\-\s]", "", regex=True)
        df["Seebeck_Value"] = df["Seebeck_Value"].astype(float)

        st.success(f"✅ Found {len(df)} entries for Seebeck coefficient.")
        st.dataframe(df)

        # Step 4: Save to SQLite database
        conn = sqlite3.connect("seebeck_data.db")
        df.to_sql("SeebeckData", conn, if_exists="replace", index=False)
        conn.commit()
        conn.close()

        with open("seebeck_data.db", "rb") as f:
            st.download_button("💾 Download SQLite DB", f, file_name="seebeck_data.db")

        # Step 5: Save to CSV for user download
        csv_data = df.to_csv(index=False)
        st.download_button("📊 Download as CSV", csv_data, file_name="seebeck_data.csv", mime="text/csv")
    else:
        st.warning("⚠️ No Seebeck coefficient data found in the uploaded PDF.")
