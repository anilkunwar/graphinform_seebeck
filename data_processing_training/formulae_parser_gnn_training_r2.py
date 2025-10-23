import os
import sqlite3
import streamlit as st
import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Span, Doc
from spacy.util import filter_spans
from spacy.matcher import PhraseMatcher
import re
import logging
import plotly.express as px
import plotly.graph_objects as go
import uuid
import psutil
from datetime import datetime
import numpy as np
from collections import Counter
import glob
from difflib import SequenceMatcher
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN

# Import PyTorch Geometric for GNN
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    from torch_geometric.loader import DataLoader
    import torch_geometric
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False
    st.error("PyTorch Geometric is required for GNN regression. Install with: `pip install torch-geometric`")
    st.stop()

# Chemical name to formula conversion (using pubchempy)
try:
    import pubchempy as pcp
    PUBCHEM_AVAILABLE = True
except ImportError:
    PUBCHEM_AVAILABLE = False
    st.warning("pubchempy not available for chemical name conversion. Install with `pip install pubchempy`")

# Define valid chemical elements
VALID_ELEMENTS = set(Element.__members__.keys())

# Invalid terms to exclude from formula detection
INVALID_TERMS = {
    'p-type', 'n-type', 'doping', 'doped', 'thermoelectric', 'material', 'the', 'and',
    'is', 'exhibits', 'type', 'based', 'sample', 'compound', 'system', 'properties',
    'references', 'acknowledgments', 'data', 'matrix', 'experimental', 'note', 'level',
    'conflict', 'result', 'captions', 'average', 'teg', 'tegs', 'marco', 'skeaf',
    'equation', 'figure', 'table', 'section', 'method', 'results', 'discussion'
}

DB_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Regex NER for formulas with fixed pattern
# -----------------------------
@Language.component("formula_ner")
def formula_ner(doc):
    formula_pattern = r'\b(?:[A-Z][a-z]?[0-9]*\.?[0-9]*)+(?::[A-Z][a-z]?[0-9]*\.?[0-9]*)?\b'
    spans = []
    for match in re.finditer(formula_pattern, doc.text):
        formula = match.group(0)
        if validate_formula(formula):
            span = doc.char_span(match.start(), match.end(), label="FORMULA")
            if span:
                spans.append(span)
    doc.ents = filter_spans(list(doc.ents) + spans)
    return doc

# -----------------------------
# Enhanced formula validation
# -----------------------------
def validate_formula(formula):
    """Validate if a string is a plausible chemical formula."""
    if not formula or not isinstance(formula, str):
        return False
    
    # Remove doping part for validation
    base_formula = re.sub(r':.+', '', formula)
    
    # Exclude non-chemical terms
    non_chemical_terms = {
        'DFT', 'TOC', 'PDOS', 'UTS', 'TEs', 'PFU', 'CNO', 'DOS', 'III', 
        'S10', 'K35', 'Ca5', 'Sb6', 'Te3', 'Te4', 'Bi2'
    }
    if base_formula.upper() in non_chemical_terms:
        return False
    
    # Skip short or invalid patterns
    if len(base_formula) <= 2 or re.match(r'^[A-Z](?:-[A-Z]|\.\d+|)$', base_formula):
        return False
    
    # Validate with pymatgen
    try:
        comp = Composition(base_formula)
        if not comp.valid:
            return False
        elements = [el.symbol for el in comp.elements]
        # Ensure at least two atoms for graph construction
        total_atoms = sum(comp.get_el_amt_dict().values())
        if total_atoms < 2:
            return False
        return all(el in VALID_ELEMENTS for el in elements)
    except Exception:
        return False

# -----------------------------
# Chemical name to formula conversion
# -----------------------------
def convert_name_to_formula(name):
    if not PUBCHEM_AVAILABLE:
        return None
    try:
        compounds = pcp.get_compounds(name, 'name')
        if compounds:
            return compounds[0].molecular_formula
        return None
    except:
        return None

# -----------------------------
# Attention-based formula scoring
# -----------------------------
def score_formula_context(formula, text, synonyms):
    """Score a formula based on its context to determine if it's a valid chemical formula."""
    score = 0.0
    context_window = 100
    start_idx = max(0, text.lower().find(formula.lower()) - context_window)
    end_idx = min(len(text), text.lower().find(formula.lower()) + len(formula) + context_window)
    context = text[start_idx:end_idx].lower()
    
    positive_terms = ['thermoelectric', 'seebeck', 'seebeck coefficient', 'thermopower', 'material', 'compound', 'semiconductor']
    positive_terms += [syn for syn_list in synonyms.values() for syn in syn_list]
    common_materials = ['Bi2Te3', 'PbTe', 'SnSe', 'CoSb3', 'SiGe', 'Skutterudite', 'Half-Heusler']
    
    for term in positive_terms + common_materials:
        if term.lower() in context:
            score += 0.2
    
    negative_terms = ['figure', 'table', 'references', 'acknowledgments', 'section', 'equation']
    for term in negative_terms:
        if term.lower() in context:
            score -= 0.3
    
    return max(0.0, min(score, 1.0))

# -----------------------------
# Material matcher with synonyms
# -----------------------------
def build_material_matcher(nlp, synonyms):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for canonical, variants in synonyms.items():
        patterns = [nlp.make_doc(v) for v in variants]
        matcher.add(canonical, patterns)
    return matcher

@Language.component("material_matcher")
def material_matcher(doc):
    matcher = doc._.material_matcher
    matches = matcher(doc)
    spans = []
    for match_id, start, end in matches:
        canonical = doc.vocab.strings[match_id]
        span = Span(doc, start, end, label="MATERIAL_TYPE")
        span._.norm = canonical
        spans.append(span)
    doc.ents = filter_spans(list(doc.ents) + spans)
    return doc

# -----------------------------
# Quantitative NER for Seebeck values with unit handling
# -----------------------------
@Language.component("seebeck_ner")
def seebeck_ner(doc):
    # Expanded regex for numbers with various unit formats
    seebeck_pattern = r'\b([-+]?\d+(?:\.\d+)?(?:\s*×\s*10\s*[\^]?\s*-?\d+)?)\s*(?:μV/K|µV/K|μV·K⁻¹|µV·K⁻¹|uV/K|microV/K|microvolt per kelvin|mV/K|V/K|microvolt/K|mv/K|v/K)\b'
    spans = []
    for match in re.finditer(seebeck_pattern, doc.text, re.IGNORECASE):
        value_str = match.group(1)
        unit = match.group(0).replace(value_str, '').strip().lower()
        try:
            value = float(value_str)
            # Unit conversion to μV/K
            if 'mv' in unit:
                value *= 1000
            elif 'v' in unit:
                value *= 1e6
            if -500 <= value <= 500:
                span = doc.char_span(match.start(), match.end(), label="SEEBECK_VALUE")
                if span:
                    span._.seebeck_value = value  # Store standardized value in extension
                    spans.append(span)
        except ValueError:
            continue
    doc.ents = filter_spans(list(doc.ents) + spans)
    return doc

# Set extension for standardized Seebeck value
if not Span.has_extension("seebeck_value"):
    Span.set_extension("seebeck_value", default=None)

# -----------------------------
# Load spaCy model with Seebeck NER
# -----------------------------
def load_spacy_model(synonyms):
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except Exception as e:
        st.error(f"Failed to load spaCy: {e}. Install: `python -m spacy download en_core_web_sm`")
        st.stop()
    
    nlp.add_pipe("formula_ner", last=True)
    nlp.add_pipe("seebeck_ner", last=True)
    matcher = build_material_matcher(nlp, synonyms)
    nlp.add_pipe("material_matcher", last=True)
    
    if not Doc.has_extension("material_matcher"):
        Doc.set_extension("material_matcher", default=None)
    Doc.set_extension("material_matcher", default=matcher, force=True)
    
    if not Span.has_extension("norm"):
        Span.set_extension("norm", default=None)
    
    return nlp

# -----------------------------
# Link formulas to Seebeck value with dependency parsing
# -----------------------------
def link_formula_to_seebeck(doc):
    formulas = [(ent, score_formula_context(ent.text, doc.text, st.session_state.synonyms)) 
                for ent in doc.ents if ent.label_ == "FORMULA"]
    formulas = [f for f, score in formulas if score > 0.3]
    seebeck_values = [ent for ent in doc.ents if ent.label_ == "SEEBECK_VALUE"]
    pairs = []
    for s in seebeck_values:
        # Use dependency parsing to find associated material
        for token in s.root.subtree:
            if token.ent_type_ == "FORMULA" or token.ent_type_ == "MATERIAL_TYPE":
                material_text = token.text
                standardized_material = standardize_material_formula(material_text)
                if not standardized_material:
                    standardized_material = convert_name_to_formula(material_text)
                if standardized_material:
                    pairs.append({
                        "Formula": standardized_material,
                        "Seebeck_Value": s._.seebeck_value  # Standardized value
                    })
                    break
        else:
            # Fallback to nearest distance if no dep link
            nearest_formula = None
            min_distance = float("inf")
            for f in formulas:
                distance = abs(f.start_char - s.start_char)
                if distance < min_distance:
                    min_distance = distance
                    nearest_formula = f
            if nearest_formula:
                pairs.append({
                    "Formula": nearest_formula.text,
                    "Seebeck_Value": s._.seebeck_value
                })
    return pairs

# -----------------------------
# Statistical Tools: PMI Calculation
# -----------------------------
def calculate_pmi(texts, term1, term2):
    """Calculate Pointwise Mutual Information for two terms in a list of texts."""
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    if term1 not in feature_names or term2 not in feature_names:
        return 0.0
    
    term1_idx = np.where(feature_names == term1)[0][0]
    term2_idx = np.where(feature_names == term2)[0][0]
    
    p_xy = np.sum((X[:, term1_idx].toarray() * X[:, term2_idx].toarray()) > 0) / len(texts)
    p_x = np.mean(X[:, term1_idx].toarray() > 0)
    p_y = np.mean(X[:, term2_idx].toarray() > 0)
    
    if p_xy == 0 or p_x == 0 or p_y == 0:
        return 0.0
    
    pmi = np.log2(p_xy / (p_x * p_y))
    return max(0, pmi)  # PPMI

# -----------------------------
# TF-IDF for term importance
# -----------------------------
def calculate_tf_idf(texts, terms):
    """Calculate TF-IDF for given terms in texts."""
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    tf_idf_scores = {}
    for term in terms:
        if term in feature_names:
            idx = np.where(feature_names == term)[0][0]
            tf_idf_scores[term] = X[:, idx].mean()
    return tf_idf_scores

# -----------------------------
# Extract Seebeck values
# -----------------------------
def extract_seebeck_values(db_file, preserve_stoichiometry=False, year_range=None):
    try:
        update_log("Starting Seebeck value extraction with advanced NER")
        update_progress("Connecting to database...")
        
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers'")
        if not cursor.fetchone():
            update_log("Database does not contain 'papers' table")
            st.session_state.error_summary.append("Database does not contain 'papers' table")
            conn.close()
            return pd.DataFrame(columns=["paper_id", "title", "material", "seebeck", "context"])
        
        cursor.execute("PRAGMA table_info(papers)")
        columns = {col[1].lower() for col in cursor.fetchall()}
        required_columns = {'id', 'title'}
        if not required_columns.issubset(columns):
            missing = required_columns - columns
            update_log(f"Missing required columns: {missing}")
            st.session_state.error_summary.append(f"Missing required columns: {missing}")
            conn.close()
            return pd.DataFrame(columns=["paper_id", "title", "material", "seebeck", "context"])
        
        text_column = detect_text_column(conn)
        if not text_column:
            st.session_state.error_summary.append("No text column (content, text, abstract, body) found in database")
            conn.close()
            return pd.DataFrame(columns=["paper_id", "title", "material", "seebeck", "context"])
        st.session_state.text_column = text_column
        
        year_column = detect_year_column(conn)
        select_columns = f"id AS paper_id, title, {text_column}"
        if year_column:
            select_columns += f", {year_column} AS year"
        
        query = f"SELECT {select_columns} FROM papers WHERE {text_column} IS NOT NULL AND {text_column} NOT LIKE 'Error%'"
        if year_column and year_range:
            query += f" AND {year_column} BETWEEN {year_range[0]} AND {year_range[1]}"
        df = pd.read_sql_query(query, conn)
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='standardized_formulas'")
        if cursor.fetchone():
            cached_df = pd.read_sql_query("SELECT material, seebeck FROM standardized_formulas", conn)
            if year_column:
                try:
                    cached_df['year'] = pd.read_sql_query("SELECT year FROM papers", conn)['year']
                except Exception as e:
                    update_log(f"Failed to load year from cached data: {str(e)}")
            if 'paper_id' not in cached_df.columns:
                cached_df['paper_id'] = pd.read_sql_query("SELECT id FROM papers", conn)['id']
            if 'title' not in cached_df.columns:
                cached_df['title'] = pd.read_sql_query("SELECT title FROM papers", conn)['title']
            if 'context' not in cached_df.columns:
                cached_df['context'] = ''
            update_log("Loaded cached standardized formulas")
            conn.close()
            return cached_df
        
        conn.close()
        
        if df.empty:
            update_log("No valid papers found for material classification")
            st.session_state.error_summary.append("No valid papers found in database")
            return pd.DataFrame(columns=["paper_id", "title", "material", "seebeck", "context"])
        
        nlp = load_spacy_model(st.session_state.synonyms)
        
        seebeck_extractions = []
        seebeck_patterns = [
            r"seebeck\s+coefficient\s+of\s+([-+]?\d+(?:\.\d+)?)\s*(?:μV/K|µV/K|μV·K⁻¹|µV·K⁻¹)",
            r"s\s*=\s*([-+]?\d+(?:\.\d+)?)\s*(?:μV/K|µV/K|μV·K⁻¹|µV·K⁻¹)",
            r"α\s*=\s*([-+]?\d+(?:\.\d+)?)\s*(?:μV/K|µV/K|μV·K⁻¹|µV·K⁻¹)",
            r"thermoelectric\s+power\s+of\s+([-+]?\d+(?:\.\d+)?)\s*(?:μV/K|µV/K|μV·K⁻¹|µV·K⁻¹)",
            r"seebeck\s*=\s*([-+]?\d+(?:\.\d+)?)\s*(?:μV/K|µV/K|μV·K⁻¹|µV·K⁻¹)",
            r"([-+]?\d+(?:\.\d+)?)\s*(?:μV/K|µV/K|μV·K⁻¹|µV·K⁻¹)\s*(?:for|of|in)\s+([A-Za-z0-9\(\)\-\s,:]+?)(?=\s|,|\.|;|:|$)"
        ]
        common_te_materials = [
            "Bi2Te3", "PbTe", "SnSe", "CoSb3", "SiGe", "Skutterudite",
            "Half-Heusler", "Clathrate", "Zn4Sb3", "Mg2Si", "Cu2Se"
        ]
        
        def chunk_text(text, max_length=200000):
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + max_length, len(text))
                if end < len(text):
                    last_period = text.rfind('.', start, end)
                    end = last_period + 1 if last_period > start else end
                chunks.append(text[start:end])
                start = end
            return chunks
        
        all_texts = df[text_column].tolist()
        progress_bar = st.progress(0)
        for i, row in df.iterrows():
            update_progress(f"Processing paper {row['paper_id']} ({i+1}/{len(df)}")
            content = row[text_column]
            chunks = chunk_text(content)
            
            for chunk_idx, chunk in enumerate(chunks):
                doc = nlp(chunk)
                formula_entities = [ent.text for ent in doc.ents if ent.label_ == "FORMULA"]
                
                # Convert chemical names to formulas
                for ent in doc.ents:
                    if ent.label_ == "MATERIAL_TYPE":
                        formula = convert_name_to_formula(ent.text)
                        if formula:
                            formula_entities.append(formula)
                
                linked_pairs = link_formula_to_seebeck(doc)
                
                for pair in linked_pairs:
                    seebeck_value = pair["Seebeck_Value"]
                    standardized_material = standardize_material_formula(pair["Formula"], preserve_stoichiometry)
                    if standardized_material:
                        classification_entry = {
                            "paper_id": row["paper_id"],
                            "title": row["title"],
                            "material": standardized_material,
                            "seebeck": seebeck_value,
                            "context": f"Found in context: {chunk[max(0, chunk.find(pair['Formula'])-50):min(len(chunk), chunk.find(pair['Formula'])+50)]}..."
                        }
                        if 'year' in row:
                            classification_entry['year'] = row['year']
                        seebeck_extractions.append(classification_entry)
                
                seebeck_materials = set()
                for pattern in seebeck_patterns:
                    matches = re.finditer(pattern, chunk, re.IGNORECASE)
                    for match in matches:
                        try:
                            value_str = match.group(1).strip()
                            value = float(value_str)
                            if -500 <= value <= 500:
                                material = match.group(2).strip() if len(match.groups()) > 1 else ""
                                if material and len(material) > 2 and material in formula_entities and validate_formula(material):
                                    standardized_material = standardize_material_formula(material, preserve_stoichiometry)
                                    if standardized_material:
                                        seebeck_materials.add((standardized_material, value, match.start()))
                        except ValueError:
                            continue
                
                # Use PMI to associate formulas with seebeck
                for formula in formula_entities:
                    pmi = calculate_pmi(all_texts, "seebeck", formula.lower())
                    if pmi > 1.0:  # Strong association threshold
                        value_match = re.search(r'([-+]?\d+(?:\.\d+)?)', chunk)
                        if value_match:
                            value = float(value_match.group(1))
                            if -500 <= value <= 500:
                                standardized_material = standardize_material_formula(formula, preserve_stoichiometry)
                                if standardized_material:
                                    seebeck_extractions.append({
                                        "paper_id": row["paper_id"],
                                        "title": row["title"],
                                        "material": standardized_material,
                                        "seebeck": value,
                                        "context": chunk
                                    })
                
                # Use TF-IDF to score terms
                tf_idf_scores = calculate_tf_idf(all_texts, ["seebeck", "thermopower", "coefficient"])
                update_log(f"TF-IDF scores: {tf_idf_scores}")
                
                doc = None
                import gc
                gc.collect()
            
            progress_value = min((i + 1) / len(df), 1.0)
            progress_bar.progress(progress_value)
        
        seebeck_df = pd.DataFrame(seebeck_extractions)
        
        if seebeck_df.empty:
            update_log("No Seebeck values extracted")
            st.session_state.error_summary.append("No Seebeck values found")
            return pd.DataFrame(columns=["paper_id", "title", "material", "seebeck", "context"])
        
        seebeck_df = seebeck_df.drop_duplicates(subset=["paper_id", "material", "seebeck"])
        seebeck_df = seebeck_df.sort_values(by=["material", "seebeck"])
        update_log(f"Cleaned and sorted DataFrame: {len(seebeck_df)} unique classifications")
        update_log(f"seebeck_df columns: {seebeck_df.columns.tolist()}")
        
        conn = sqlite3.connect(db_file)
        seebeck_df[["material", "seebeck"] + (["year"] if 'year' in seebeck_df.columns else [])].to_sql("standardized_formulas", conn, if_exists="replace", index=False)
        conn.close()
        update_log("Cached standardized formulas in database")
        
        formulas = seebeck_df["material"].tolist()
        targets = seebeck_df["seebeck"].tolist()
        model, scaler, model_files = train_gnn(formulas, targets)
        st.session_state.ann_model = model  # Keep same key for compatibility
        st.session_state.scaler = scaler
        st.session_state.model_files = model_files
        
        update_log(f"Extracted {len(seebeck_df)} Seebeck values")
        return seebeck_df
    
    except sqlite3.OperationalError as e:
        update_log(f"SQLite error: {str(e)}")
        st.session_state.error_summary.append(f"SQLite error: {str(e)}")
        return pd.DataFrame(columns=["paper_id", "title", "material", "seebeck", "context"])
    except Exception as e:
        update_log(f"Error in Seebeck extraction: {str(e)}")
        st.session_state.error_summary.append(f"Extraction error: {str(e)}")
        return pd.DataFrame(columns=["paper_id", "title", "material", "seebeck", "context"])

# -----------------------------
# Main Streamlit app
# -----------------------------
st.set_page_config(page_title="Thermoelectric Material Seebeck Tool", layout="wide")
st.title("Thermoelectric Material Seebeck Analysis Tool")
st.markdown("""
This tool extracts Seebeck coefficients from SQLite databases and allows prediction of user-input chemical formulas using NLP and GNN.

**Dependencies**:
- `pip install streamlit pandas sqlite3 spacy plotly psutil pymatgen scikit-learn joblib torch torch-geometric h5py pubchempy`
- `python -m spacy download en_core_web_sm`
""")

# Initialize session state
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "seebeck_extractions" not in st.session_state:
    st.session_state.seebeck_extractions = None
if "db_file" not in st.session_state:
    st.session_state.db_file = None
if "error_summary" not in st.session_state:
    st.session_state.error_summary = []
if "progress_log" not in st.session_state:
    st.session_state.progress_log = []
if "text_column" not in st.session_state:
    st.session_state.text_column = "content"
if "synonyms" not in st.session_state:
    st.session_state.synonyms = {
        "seebeck": ["seebeck coefficient", "positive type", "positive thermoelectric", "hole conducting"],
        "material": ["n-type", "negative type", "negative thermoelectric", "electron conducting"]
    }
if "ann_model" not in st.session_state:
    st.session_state.ann_model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "save_formats" not in st.session_state:
    st.session_state.save_formats = ["pkl", "db", "pt", "h5"]
if "model_files" not in st.session_state:
    st.session_state.model_files = {}

# Database selection
st.header("Select or Upload Database")
db_files = glob.glob(os.path.join(DB_DIR, "*.db"))
db_options = [os.path.basename(f) for f in db_files] + ["Upload a new .db file"]
db_selection = st.selectbox("Select Database", db_options, index=0, key="db_select")
uploaded_file = None
if db_selection == "Upload a new .db file":
    uploaded_file = st.file_uploader("Upload SQLite Database (.db)", type=["db"], key="db_upload")
    if uploaded_file:
        temp_db_path = os.path.join(DB_DIR, f"uploaded_{uuid.uuid4().hex}.db")
        with open(temp_db_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.db_file = temp_db_path
        update_log(f"Uploaded database saved as {temp_db_path}")
else:
    if db_selection:
        st.session_state.db_file = os.path.join(DB_DIR, db_selection)
        update_log(f"Selected database: {db_selection}")

# Database preview and validation
if st.session_state.db_file:
    try:
        conn = sqlite3.connect(st.session_state.db_file)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA table_info(papers)")
        db_columns = [col[1].lower() for col in cursor.fetchall()]
        update_log(f"Database 'papers' table columns: {db_columns}")
        
        text_column = detect_text_column(conn)
        if not text_column:
            st.error("No text column (content, text, abstract, body) found in database. Please check the database schema.")
            conn.close()
            st.stop()
        
        cursor.execute(f"SELECT COUNT(*) FROM papers WHERE {text_column} IS NOT NULL AND {text_column} NOT LIKE 'Error%'")
        paper_count = cursor.fetchone()[0]
        
        year_column = detect_year_column(conn)
        select_columns = f"id, title, {text_column}"
        if year_column:
            select_columns += f", {year_column} AS year"
        
        query = f"SELECT {select_columns} FROM papers WHERE {text_column} IS NOT NULL AND {text_column} NOT LIKE 'Error%' LIMIT 5"
        preview_data = pd.read_sql_query(query, conn)
        conn.close()
        
        st.info(f"Database contains {paper_count} valid papers.")
        
        st.subheader("Database Preview (First 5 Papers)")
        display_columns = [col for col in ["id", "title", "year"] if col in preview_data.columns]
        update_log(f"Preview data columns: {preview_data.columns.tolist()}")
        
        if text_column in preview_data.columns:
            preview_data_display = preview_data[display_columns].copy()
            preview_data_display[f"{text_column}_preview"] = preview_data[text_column].str[:100] + "..."
            st.dataframe(preview_data_display, use_container_width=True)
        else:
            st.dataframe(preview_data[display_columns], use_container_width=True)
            st.warning(f"Text column '{text_column}' not found in preview data. Available columns: {', '.join(preview_data.columns)}")
        
        if st.button("Clear Cached Formulas", key="clear_cache"):
            conn = sqlite3.connect(st.session_state.db_file)
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS standardized_formulas")
            cursor.execute("DROP TABLE IF EXISTS models")
            conn.commit()
            conn.close()
            update_log("Cleared cached standardized formulas and models")
            st.success("Cached formulas and models cleared. Run extraction again to refresh.")
    
    except sqlite3.OperationalError as e:
        st.error(f"Database error: {str(e)}")
        st.session_state.error_summary.append(f"Database error: {str(e)}")
        st.stop()
    
    # Tabs for Seebeck extraction and formula prediction
    tab1, tab2 = st.tabs(["Seebeck Extraction", "Formula Prediction"])
    
    with tab1:
        st.header("Seebeck Coefficient Extraction Analysis")
        
        with st.sidebar:
            st.subheader("Seebeck Extraction Parameters")
            material_top_n = st.slider("Number of Top Materials to Show", min_value=5, max_value=30, value=10, key="material_top_n")
            preserve_stoichiometry = st.checkbox("Preserve Exact Stoichiometry", value=False, key="preserve_stoichiometry")
            year_range = st.slider("Year Range", min_value=1980, max_value=2025, value=(2000, 2025), key="year_range")
            
            st.subheader("Model Save Formats")
            save_formats = st.multiselect(
                "Select formats to save models",
                options=["db", "pkl", "pt", "h5"],
                default=st.session_state.get('save_formats', ["pkl", "db", "pt", "h5"]),
                key="save_formats_selector"
            )
            if save_formats != st.session_state.get('save_formats', []):
                st.session_state['save_formats'] = save_formats
                update_log(f"Updated save formats to: {save_formats}")
            st.write("Models will be saved in:", ", ".join(st.session_state.save_formats) if st.session_state.save_formats else "None")
            
            st.subheader("Synonym Settings")
            with st.form("add_synonym_form"):
                st.write("➕ Add new synonym")
                synonym_text = st.text_input("Phrase (e.g. 'hole transport'):", key="synonym_text")
                synonym_type = st.selectbox("Maps to:", ["seebeck", "material"], key="synonym_type")
                submitted = st.form_submit_button("Add Synonym")
                if submitted and synonym_text.strip():
                    st.session_state.synonyms[synonym_type].append(synonym_text.strip())
                    st.success(f"Added '{synonym_text}' → {synonym_type}")
                    update_log(f"Added synonym '{synonym_text}' for {synonym_type}")
            
            st.subheader("Remove Synonym")
            with st.form("remove_synonym_form"):
                synonym_to_remove = st.selectbox(
                    "Select synonym to remove:",
                    options=sum([[f"{syn} ({typ})" for syn in synonyms] for typ, synonyms in st.session_state.synonyms.items()], []),
                    key="synonym_remove_select"
                )
                remove_submitted = st.form_submit_button("Remove Synonym")
                if remove_submitted and synonym_to_remove:
                    syn, typ = synonym_to_remove.rsplit(" (", 1)
                    typ = typ.rstrip(")")
                    if syn in st.session_state.synonyms[typ]:
                        st.session_state.synonyms[typ].remove(syn)
                        st.success(f"Removed '{syn}' from {typ}")
                        update_log(f"Removed synonym '{syn}' from {typ}")
            
            st.write("### Current synonyms:")
            st.json(st.session_state.synonyms)
            
            material_filter_options = st.session_state.get("material_filter_options", [])
            material_filter = st.multiselect("Filter Materials", options=material_filter_options, 
                                           placeholder="Select materials after extraction", key="material_filter")
        
        if st.button("Extract Seebeck Values", key="extract_seebeck"):
            st.session_state.error_summary = []
            st.session_state.progress_log = []
            with st.spinner("Extracting Seebeck values..."):
                seebeck_df = extract_seebeck_values(st.session_state.db_file, preserve_stoichiometry, year_range)
                st.session_state.seebeck_extractions = seebeck_df
                
                if not seebeck_df.empty:
                    st.session_state.material_filter_options = sorted(seebeck_df["material"].unique())
            
            if seebeck_df.empty:
                st.warning("No Seebeck values found. Check logs for details.")
                if st.session_state.error_summary:
                    st.error("Errors encountered:\n- " + "\n- ".join(set(st.session_state.error_summary)))
            else:
                st.success(f"Extracted {len(seebeck_df)} unique Seebeck values!")
                
                filtered_df = seebeck_df if not material_filter else seebeck_df[seebeck_df["material"].isin(material_filter)]
                
                if material_filter and not seebeck_df["material"].isin(material_filter).any():
                    update_log("Material filter resulted in empty DataFrame")
                    st.warning("Selected materials not found in extracted data. Showing all values.")
                    filtered_df = seebeck_df
                
                display_columns = ["paper_id", "title", "material", "seebeck", "context"]
                if 'year' in filtered_df.columns:
                    display_columns.insert(2, "year")
                
                available_columns = [col for col in display_columns if col in filtered_df.columns]
                if len(available_columns) < len(display_columns):
                    missing_columns = [col for col in display_columns if col not in filtered_df.columns]
                    update_log(f"Missing columns in filtered_df: {missing_columns}")
                    st.warning(f"Some expected columns are missing: {', '.join(missing_columns)}. Displaying available columns: {', '.join(available_columns)}")
                
                st.subheader("Extraction Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Extractions", len(filtered_df))
                with col2:
                    avg_seebeck = filtered_df["seebeck"].mean()
                    st.metric("Average Seebeck", f"{avg_seebeck:.2f} μV/K")
                with col3:
                    std_seebeck = filtered_df["seebeck"].std()
                    st.metric("Std Dev Seebeck", f"{std_seebeck:.2f} μV/K")
                
                st.subheader("Visualizations")
                fig_bar, fig_hist, fig_timeline, fig_heatmap, fig_sunburst = plot_seebeck_values(filtered_df, material_top_n, year_range)
                
                if fig_bar:
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.warning("No data available for bar chart.")
                
                col1, col2 = st.columns(2)
                with col1:
                    if fig_hist:
                        st.plotly_chart(fig_hist, use_container_width=True)
                    else:
                        st.warning("No data available for histogram.")
                with col2:
                    if fig_timeline:
                        st.plotly_chart(fig_timeline, use_container_width=True)
                    else:
                        st.warning("No data available for timeline chart.")
                
                if fig_heatmap:
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.warning("No data available for co-occurrence heatmap.")
                
                if fig_sunburst:
                    st.plotly_chart(fig_sunburst, use_container_width=True)
                else:
                    st.warning("No data available for sunburst chart.")
                
                st.subheader("Extracted Seebeck Values")
                update_log(f"Attempting to display columns: {available_columns}")
                if available_columns:
                    st.dataframe(
                        filtered_df[available_columns].head(100),
                        use_container_width=True
                    )
                else:
                    st.error("No valid columns available to display values.")
                
                csv_df = filtered_df[["material", "seebeck"] + (["year"] if 'year' in filtered_df.columns else [])].rename(
                    columns={"material": "Formula", "seebeck": "Seebeck Coefficient (μV/K)", "year": "Year"}
                )
                material_csv = csv_df.to_csv(index=False)
                st.download_button(
                    "Download Seebeck Values CSV", 
                    material_csv, 
                    "seebeck_values_via_nlp.csv", 
                    "text/csv", 
                    key="download_seebeck"
                )
                
                if hasattr(st.session_state, 'model_files'):
                    st.subheader("Download Saved Models")
                    for model_file, file_path in st.session_state.model_files.items():
                        try:
                            with open(file_path, 'rb') as f:
                                st.download_button(
                                    f"Download {model_file}",
                                    f,
                                    model_file,
                                    key=f"download_{model_file}"
                                )
                        except Exception as e:
                            st.error(f"Failed to provide download for {model_file}: {str(e)}")
                
                st.subheader("Extraction Progress")
                progress_log_display = "\n".join(st.session_state.progress_log) if st.session_state.progress_log else "No progress messages yet."
                st.text(progress_log_display)
        
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="seebeck_logs")
    
    with tab2:
        st.header("Seebeck Prediction")
        st.markdown("""
        Enter a chemical formula or upload a CSV file with formulas to predict their Seebeck coefficient.
        Predictions are based on extracted data or GNN for unseen formulas.
        **Note**: Run Seebeck Extraction Analysis first to populate the data and train the GNN.
        """)
        
        with st.sidebar:
            st.subheader("Prediction Parameters")
            prediction_mode = st.radio("Input Mode", ["Single Formula", "Batch CSV Upload"], key="prediction_mode")
            fuzzy_match = st.checkbox("Enable Fuzzy Matching", value=False, key="fuzzy_match")
        
        if prediction_mode == "Single Formula":
            formula_input = st.text_input("Enter Chemical Formula (e.g., Bi2Te3, PbTe)", key="formula_input")
            corrected_formula = st.text_input("Corrected Formula (optional)", value=formula_input, key="corrected_formula")
            if st.button("Predict Seebeck", key="predict_seebeck"):
                if not formula_input:
                    st.error("Please enter a chemical formula.")
                else:
                    with st.spinner(f"Predicting Seebeck for '{corrected_formula}'..."):
                        result, error, similar_formula = predict_seebeck(corrected_formula, st.session_state.seebeck_extractions, fuzzy_match)
                        if error:
                            st.error(error)
                            if similar_formula:
                                st.warning(f"Suggested similar formula: {similar_formula}")
                                if st.button(f"Predict Suggested Formula: {similar_formula}", key="predict_similar"):
                                    result, error, _ = predict_seebeck(similar_formula, st.session_state.seebeck_extractions, fuzzy_match)
                                    if error:
                                        st.error(error)
                                    else:
                                        st.success(f"Formula: **{result['formula']}**")
                                        st.write(f"Seebeck: **{result['seebeck']:.2f}** μV/K (Std: {result['std']:.2f})")
                                        if result['count'] > 0:
                                            st.write(f"Found in {result['count']} paper(s): {', '.join(result['paper_ids'])}")
                                            st.write("Context Snippets:")
                                            for i, context in enumerate(result['contexts'][:5], 1):
                                                st.write(f"{i}. {context}")
                                        else:
                                            st.write("Prediction based on GNN.")
                                        st.write("All Values:", result['all_values'])
                        else:
                            st.success(f"Formula: **{result['formula']}**")
                            st.write(f"Seebeck: **{result['seebeck']:.2f}** μV/K (Std: {result['std']:.2f})")
                            if result['count'] > 0:
                                st.write(f"Found in {result['count']} paper(s): {', '.join(result['paper_ids'])}")
                                st.write("Context Snippets:")
                                for i, context in enumerate(result['contexts'][:5], 1):
                                    st.write(f"{i}. {context}")
                            else:
                                st.write("Prediction based on GNN.")
                            st.write("All Values:", result['all_values'])
        
        else:
            uploaded_csv = st.file_uploader("Upload CSV with Formulas (column: 'formula')", type=["csv"], key="seebeck_csv")
            if uploaded_csv and st.button("Predict Batch Seebeck", key="predict_batch"):
                with st.spinner("Predicting batch Seebeck..."):
                    formulas_df = pd.read_csv(uploaded_csv)
                    if 'formula' not in formulas_df.columns:
                        st.error("CSV must contain a 'formula' column.")
                    else:
                        formulas = formulas_df['formula'].dropna().tolist()
                        results, errors, suggestions = batch_predict_seebeck(formulas, st.session_state.seebeck_extractions, fuzzy_match)
                        
                        if errors:
                            st.error("Errors encountered:\n- " + "\n- ".join(set(errors)))
                            if suggestions:
                                st.warning("Suggested corrections for some formulas:")
                                for formula, suggestion in suggestions:
                                    st.write(f"{formula} -> {suggestion}")
                        
                        if results:
                            batch_df = pd.DataFrame([{
                                "Formula": r["formula"],
                                "Seebeck Coefficient (μV/K)": f"{r['seebeck']:.2f}",
                                "Paper Count": r["count"],
                                "Paper IDs": ", ".join(r["paper_ids"])
                            } for r in results])
                            st.subheader("Batch Prediction Results")
                            st.dataframe(batch_df, use_container_width=True)
                            
                            batch_csv = batch_df.to_csv(index=False)
                            st.download_button(
                                "Download Batch Prediction Results", 
                                batch_csv, 
                                "batch_seebeck_predictions.csv", 
                                "text/csv", 
                                key="download_batch"
                            )
        
        st.text_area("Logs", "\n".join(st.session_state.log_buffer), height=150, key="prediction_logs")
else:
    st.warning("Select or upload a database file.")
