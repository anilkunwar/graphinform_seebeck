import re
import streamlit as st
from quantulum3 import parser as qparser
import spacy
from chemdataextractor import Document

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# --- Define patterns safely ---
num_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
units_pattern = r"(?:µV\s*/\s*K|μV\s*/\s*K|uV\s*/\s*K|microvolt[s]?\s*(?:per|/)\s*kelvin|µV·K⁻¹|µV K-1|μV K-1)"

# Safer value pattern
value_regex = re.compile(
    rf"""
    (?P<number_full>                 # Full numeric match group
        (?:±|\+/-)?\s*               # optional ± prefix
        {num_pattern}                # number
        (?:\s*(?:to|–|-|—)\s*{num_pattern})?   # optional range
        (?:\s*(?:±|\+/-)\s*{num_pattern})?     # optional uncertainty
    )
    \s*
    (?P<unit>{units_pattern})        # capture the unit
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

# --- Helper: Extract materials and Seebeck values from text ---
def extract_materials_and_seebeck(text):
    # Resolve coreference (if available)
    try:
        from transformers import pipeline
        coref = pipeline("coreference-resolution", model="allenai/coref-spanbert-large")
        text = coref(text)["resolved_text"]
    except Exception:
        pass  # skip if model not available

    doc = nlp(text)
    results = []

    # ChemDataExtractor for materials
    cdoc = Document(text)
    materials = [cem.text for cem in cdoc.cems]
    material_set = set(materials)

    # Search for Seebeck phrases sentence-wise
    for sent in doc.sents:
        sent_text = sent.text
        if re.search(r"(seebeck|thermoelectric power|S[-\s]?value)", sent_text, re.IGNORECASE):
            # Quantulum3 parsing (robust unit/value detection)
            quants = qparser.parse(sent_text)
            for q in quants:
                if q.unit and any(u in str(q.unit).lower() for u in ["volt", "µv", "μv"]) and "kelvin" in str(q.unit).lower():
                    # Find nearest material mention
                    nearest_mat = None
                    min_dist = float("inf")
                    for mat in material_set:
                        idx_mat = sent_text.lower().find(mat.lower())
                        if idx_mat != -1:
                            dist = abs(idx_mat - sent_text.lower().find(str(q.value)))
                            if dist < min_dist:
                                nearest_mat = mat
                                min_dist = dist

                    results.append({
                        "Material": nearest_mat if nearest_mat else "Unknown",
                        "Value": q.value,
                        "Unit": str(q.unit),
                        "Context": sent_text.strip(),
                    })

            # Fallback regex extraction
            for m in value_regex.finditer(sent_text):
                val = m.group("number_full")
                unit = m.group("unit")
                nearest_mat = None
                for mat in material_set:
                    if mat in sent_text:
                        nearest_mat = mat
                        break
                results.append({
                    "Material": nearest_mat if nearest_mat else "Unknown",
                    "Value": val,
                    "Unit": unit,
                    "Context": sent_text.strip(),
                })
    return results

# --- Streamlit UI ---
st.title("Seebeck Coefficient Extractor (Upgraded v2)")
uploaded = st.file_uploader("Upload a research paper (PDF or TXT)", type=["pdf", "txt"])
if uploaded:
    import pdfplumber
    text = ""
    if uploaded.name.lower().endswith(".pdf"):
        with pdfplumber.open(uploaded) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    else:
        text = uploaded.read().decode("utf-8", errors="ignore")

    with st.spinner("Extracting data..."):
        extracted = extract_materials_and_seebeck(text)

    if extracted:
        st.success(f"✅ Found {len(extracted)} Seebeck data entries")
        st.dataframe(extracted)
    else:
        st.warning("No Seebeck data found in this document.")
