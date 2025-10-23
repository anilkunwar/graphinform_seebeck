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

# --- PyTorch Geometric Imports ---
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    from torch_geometric.loader import DataLoader
    import torch_geometric
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False
    # In a real Streamlit app, this would be an error, but kept for code structure
    # st.error("PyTorch Geometric is required for GNN. Install with: `pip install torch-geometric`")
    print("Warning: PyTorch Geometric not found. GNN functions will fail.")

# --- Global/Placeholder Setup (Simulating Streamlit environment) ---
DB_DIR = "data"
if 'error_summary' not in st.session_state:
     st.session_state.error_summary = []
if 'synonyms' not in st.session_state:
     st.session_state.synonyms = {}
if 'text_column' not in st.session_state:
     st.session_state.text_column = "content"
if 'gnn_regressor_model' not in st.session_state:
     st.session_state.gnn_regressor_model = None
if 'seebeck_scaler' not in st.session_state:
     st.session_state.seebeck_scaler = None

def update_log(message):
    # Placeholder for the original function
    # print(f"LOG: {message}")
    pass 

def update_progress(message):
    # Placeholder for the original function
    # print(f"PROGRESS: {message}")
    pass 

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

# -----------------------------
# Formula and Context NLP/NER (KEPT)
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

def validate_formula(formula):
    """Validate if a string is a plausible chemical formula."""
    if not formula or not isinstance(formula, str):
        return False
    base_formula = re.sub(r':.+', '', formula)
    non_chemical_terms = {
        'DFT', 'TOC', 'PDOS', 'UTS', 'TEs', 'PFU', 'CNO', 'DOS', 'III', 
        'S10', 'K35', 'Ca5', 'Sb6', 'Te3', 'Te4', 'Bi2', 'I', 'II', 'IV'
    }
    if base_formula.upper() in non_chemical_terms:
        return False
    if len(base_formula) <= 2 or re.match(r'^[A-Z](?:-[A-Z]|\.\d+|)$', base_formula):
        return False
    try:
        comp = Composition(base_formula)
        if not comp.valid:
            return False
        elements = [el.symbol for el in comp.elements]
        return all(el in VALID_ELEMENTS for el in elements)
    except Exception:
        return False

def score_formula_context(formula, text, synonyms):
    """Score a formula based on its context to determine if it's a valid chemical formula and related to TE/Seebeck."""
    score = 0.0
    context_window = 100
    formula_lower = formula.lower()
    match_start = text.lower().find(formula_lower)
    if match_start == -1: return 0.0

    start_idx = max(0, match_start - context_window)
    end_idx = min(len(text), match_start + len(formula) + context_window)
    context = text[start_idx:end_idx].lower()
    
    positive_terms = ['thermoelectric', 'seebeck coefficient', 'seebeck', 's', 'material', 'compound', 'semiconductor', 'figure of merit', 'zt']
    for term in positive_terms:
        if term in context:
            score += 0.2
    
    if re.search(r'(seebeck|s)\s+(coefficient|value|of)\s+.*?' + re.escape(formula_lower), context) or \
       re.search(re.escape(formula_lower) + r'\s+.*?(seebeck|s)\s+(coefficient|value|of)', context):
        score += 0.5

    negative_terms = ['figure', 'table', 'references', 'acknowledgments', 'section', 'equation', 'units']
    for term in negative_terms:
        if term in context:
            score -= 0.3
    
    return max(0.0, min(score, 1.0))

def build_material_matcher(nlp, synonyms):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for canonical, variants in synonyms.items():
        patterns = [nlp.make_doc(v) for v in variants]
        matcher.add(canonical, patterns)
    return matcher

@Language.component("material_matcher")
def material_matcher(doc):
    # This is mainly kept for compatibility with the original structure, though less crucial for Seebeck extraction
    if not Doc.has_extension("material_matcher"): return doc
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

def load_spacy_model(synonyms):
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except Exception as e:
         # Simplified error handling for the full code block
         raise Exception(f"Failed to load spaCy: {e}")
    
    nlp.add_pipe("formula_ner", last=True)
    matcher = build_material_matcher(nlp, synonyms)
    nlp.add_pipe("material_matcher", last=True)
    
    if not Doc.has_extension("material_matcher"):
        Doc.set_extension("material_matcher", default=None)
    Doc.set_extension("material_matcher", default=matcher, force=True)
    
    if not Span.has_extension("norm"):
        Span.set_extension("norm", default=None)
    
    return nlp

def standardize_material_formula(formula, preserve_stoichiometry=False, canonical_order=True):
    # ... (Standardization logic remains the same) ...
    if not formula or not isinstance(formula, str): return None
    formula = re.sub(r'\s+', '', formula)
    formula = re.sub(r'[\[\]\{\}]', '', formula)
    if not validate_formula(formula): return None
    
    doping_pattern = r'(.+?)(?::|doped\s+)([A-Za-z0-9,\.]+)'
    doping_match = re.match(doping_pattern, formula, re.IGNORECASE)
    dopants = None
    if doping_match:
        base_formula, dopants = doping_match.groups()
        formula = base_formula.strip()
        dopants = dopants.split(',')
    
    try:
        comp = Composition(formula)
        if not comp.valid: return None
        
        if preserve_stoichiometry:
            el_amt_dict = comp.get_el_amt_dict()
            standardized_formula = ''.join(
                f"{el}{amt:.2f}" if amt != int(amt) else f"{el}{int(amt)}"
                for el, amt in (sorted(el_amt_dict.items()) if canonical_order else el_amt_dict.items())
            )
        else:
            standardized_formula = comp.hill_formula 
        
        if dopants:
            valid_dopants = []
            for dopant in dopants:
                if not validate_formula(dopant): continue
                try:
                    dopant_comp = Composition(dopant.strip())
                    valid_dopants.append(dopant_comp.reduced_formula)
                except Exception:
                    continue
            if valid_dopants:
                standardized_formula = f"{standardized_formula}:{','.join(valid_dopants)}"
        
        return standardized_formula
    except Exception:
        return None

# -----------------------------
# Seebeck Extraction Logic (KEPT/SLIGHTLY REFINED)
# -----------------------------
def find_seebeck_mentions(doc):
    """Uses regex to find numerical values associated with the Seebeck coefficient."""
    
    # Regex to find a number and typical Seebeck units (µV/K or V/K) within proximity to a keyword.
    # We search a window around 'seebeck' or 'S' for a value and unit.
    value_unit_regex = r'([\-\+]?\s*\d{1,4}(?:[.,]\d{1,3})?)\s*([uμ]?[Vv](?:[\s/]\s*|K[\-]?1))'
    
    matches = []
    
    # Search for 'Seebeck' or 'S' mentions
    seebeck_keywords = re.finditer(r'seebeck\s+coefficient|\b[Ss]\b\s+value|\b[Ss][\s\-][aA]t|\b[Ss]\s*[\(\-=]\s*[\-\+]?\s*\d', doc.text, re.IGNORECASE)
    
    for keyword_match in seebeck_keywords:
        start = max(0, keyword_match.start() - 50)
        end = min(len(doc.text), keyword_match.end() + 100)
        context = doc.text[start:end]
        
        # Find the number and unit within the context window
        value_unit_match = re.search(value_unit_regex, context)
        
        if value_unit_match:
            # Clean and normalize value/unit
            value = value_unit_match.group(1).replace(' ', '').replace(',', '.')
            unit = value_unit_match.group(2).replace(' ', '')
            
            # Recalculate span indices within the original document
            value_start = start + value_unit_match.start(1)
            value_end = start + value_unit_match.end(2)
            
            span = doc.char_span(value_start, value_end, label="SEEBECK_VALUE")
            if span:
                matches.append((
                    span, 
                    float(value), 
                    unit, 
                    span.sent.text # Use the sentence for local context
                ))
                
    return matches

def link_formula_to_seebeck(doc, seebeck_matches):
    """Links extracted Seebeck values to the nearest high-scoring chemical formula."""
    
    # 1. Get high-scoring formulas
    formulas = [(ent, score_formula_context(ent.text, doc.text, st.session_state.synonyms)) 
                for ent in doc.ents if ent.label_ == "FORMULA"]
    high_score_formulas = [f for f, score in formulas if score > 0.4] 

    if not high_score_formulas or not seebeck_matches:
        return []

    results = []
    
    for seebeck_match in seebeck_matches:
        seebeck_span, seebeck_value, seebeck_unit, context = seebeck_match
        
        nearest_formula = None
        min_distance = float("inf")
        
        # 2. Find the nearest formula to the Seebeck mention
        for f in high_score_formulas:
            distance = abs(f.start_char - seebeck_span.start_char)
            if distance < min_distance:
                min_distance = distance
                nearest_formula = f
        
        # 3. Apply a maximum distance constraint
        # Max distance of 150 characters to ensure context relevance
        if nearest_formula and min_distance < 150:
            results.append({
                "Formula_Span": nearest_formula,
                "Seebeck_Value": seebeck_value,
                "Seebeck_Unit": seebeck_unit,
                "Context": context
            })
            
    return results

def detect_text_column(conn):
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(papers)")
    columns = [col[1].lower() for col in cursor.fetchall()]
    for col in ['content', 'text', 'abstract', 'body']:
        if col in columns: return col
    return None

def detect_year_column(conn):
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(papers)")
    columns = [col[1].lower() for col in cursor.fetchall()]
    for col in ['year', 'pub_year', 'publication_year']:
        if col in columns: return col
    return None

# -----------------------------
# GNN MODELING FOR REGRESSION
# -----------------------------

class GNNRegressor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        super(GNNRegressor, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # Output_dim is 1 for a continuous regression value (Seebeck coefficient)
        self.fc = nn.Linear(hidden_dim, 1) 

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global mean pooling to get a graph-level feature vector
        x = torch_geometric.nn.global_mean_pool(x, data.batch) 
        
        x = self.fc(x)
        return x.squeeze(1) # Ensure output is a vector of predictions

def featurize_formulas(formulas, seebeck_targets=None):
    """Convert formulas to graph data structures for GNN regression."""
    if not PYTORCH_GEOMETRIC_AVAILABLE:
        update_log("PyTorch Geometric is not available for featurization.")
        return [], [], [] if seebeck_targets is not None else None

    data_list = []
    valid_formulas = []
    valid_targets = [] if seebeck_targets is not None else None

    element_properties = { 
        el.symbol: [
            float(el.Z or 0), float(el.X or 0), float(el.group or 0),
            float(el.row or 0), float(el.atomic_mass or 0)
        ] for el in Element
    }

    for i, formula in enumerate(formulas):
        if not validate_formula(formula): continue

        try:
            # Structure creation and graph building
            comp = Composition(formula)
            el_amt_dict = comp.get_el_amt_dict()
            el_amt_dict = {k: max(1, round(v)) for k, v in el_amt_dict.items()} 
            total_atoms = sum(el_amt_dict.values())
            if total_atoms < 2: continue

            species = []
            frac_coords = []
            pos = 0
            for el, amt in el_amt_dict.items():
                for _ in range(int(amt)):
                    species.append(el)
                    frac_coords.append([pos * 0.1, 0, 0])
                    pos += 1

            if len(species) < 2: continue
            lattice = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
            structure = Structure(lattice, species, frac_coords, coords_are_cartesian=False)

            strategy = MinimumDistanceNN(cutoff=10.0) 
            sg = StructureGraph.with_local_env_strategy(structure, strategy)

            # Node features
            node_features = []
            for site in structure:
                el = site.specie.symbol
                props = element_properties.get(el, [0.0] * 5)
                node_features.append(props)
            node_features = torch.tensor(node_features, dtype=torch.float32)

            # Edge indices
            edge_index = []
            edge_weights = []
            adjacency = list(sg.graph.adjacency())
            if not adjacency or len(structure) < 2:
                for i in range(len(structure)):
                    for j in range(i + 1, len(structure)):
                        edge_index.append([i, j])
                        edge_index.append([j, i])
                        edge_weights.append(1.0)
            else:
                for i, neighbor_dict in enumerate(adjacency):
                    for neighbor_idx, data in neighbor_dict[1].items():
                        edge_index.append([i, neighbor_idx])
                        edge_weights.append(data.get('weight', 1.0))

            if not edge_index: continue

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

            # Create Data object - TARGET 'y' IS NOW A FLOAT TENSOR
            target_tensor = torch.tensor([seebeck_targets[i]], dtype=torch.float32) if seebeck_targets is not None else None
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_weights.unsqueeze(-1),
                y=target_tensor
            )

            data_list.append(data)
            valid_formulas.append(formula)
            if seebeck_targets is not None:
                valid_targets.append(seebeck_targets[i])

        except Exception as e:
            update_log(f"Failed to featurize formula '{formula}': {str(e)}")
            continue

    return data_list, valid_formulas, valid_targets

def train_gnn_regressor(formulas, seebeck_targets):
    """Trains the GNN model for Seebeck coefficient regression."""
    if not PYTORCH_GEOMETRIC_AVAILABLE:
        return None, None, {}

    if not formulas or not seebeck_targets:
        update_log("No valid data for GNN training")
        return None, None, {}

    # 1. Prepare and scale data
    scaler = StandardScaler()
    scaled_targets = scaler.fit_transform(np.array(seebeck_targets).reshape(-1, 1)).flatten()
    data_list, valid_formulas, _ = featurize_formulas(formulas, scaled_targets)
    
    if not data_list:
        update_log("No valid graph data for GNN training")
        return None, None, {}

    # Update targets in Data objects with scaled values
    for i, data in enumerate(data_list):
        data.y = torch.tensor([scaled_targets[i]], dtype=torch.float32)

    loader = DataLoader(data_list, batch_size=32, shuffle=True)

    # 2. Initialize Model
    model = GNNRegressor(input_dim=5, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss() # Mean Squared Error for regression

    # 3. Training loop
    model.train()
    for epoch in range(100):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        update_log(f"Epoch {epoch+1}, Loss (MSE): {total_loss/len(loader):.4f}")

    # 4. Save Model and Scaler
    model_files = {}
    try:
        # Save scaler
        scaler_path = os.path.join(DB_DIR, "seebeck_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        model_files["seebeck_scaler.pkl"] = scaler_path
        
        # Save PyTorch model state dict
        model_path = os.path.join(DB_DIR, "gnn_regressor.pt")
        torch.save(model.state_dict(), model_path)
        model_files["gnn_regressor.pt"] = model_path
        update_log("Saved GNN Regressor and Scaler")
    except Exception as e:
        update_log(f"Failed to save GNN/Scaler: {str(e)}")

    return model, scaler, model_files

def predict_seebeck_gnn(formula, gnn_model, scaler):
    """Predicts the Seebeck coefficient for a single user-supplied formula."""
    if gnn_model is None or scaler is None or not PYTORCH_GEOMETRIC_AVAILABLE:
        return None, "GNN Model/Scaler not loaded or PyG not available."

    normalized_formula = standardize_material_formula(formula)
    if not normalized_formula:
        return None, f"'{formula}' is not a valid chemical formula for prediction."
    
    data_list, _, _ = featurize_formulas([normalized_formula])
    if not data_list:
        return None, f"Could not featurize formula '{normalized_formula}' for prediction."

    try:
        gnn_model.eval()
        data = data_list[0]
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long) 
        
        with torch.no_grad():
            # 1. Get the scaled prediction
            scaled_prediction_tensor = gnn_model(data)
            scaled_prediction = scaled_prediction_tensor.item()
            
            # 2. Inverse transform to the real Seebeck value (µV/K)
            prediction_array = np.array([[scaled_prediction]])
            real_prediction = scaler.inverse_transform(prediction_array)[0][0]
            
            return real_prediction, None
            
    except Exception as e:
        update_log(f"GNN prediction failed: {str(e)}")
        return None, f"Prediction error: {str(e)}"

# -----------------------------
# Main Extraction Function (MODIFIED)
# -----------------------------
def extract_seebeck_values(db_file, preserve_stoichiometry=False, year_range=None):
    """
    Extracts Seebeck values from the database and prepares data for GNN training.
    Returns: extracted_df, training_formulas, training_targets
    """
    try:
        update_log("Starting Seebeck coefficient extraction with NER and Regex")
        update_progress("Connecting to database...")
        
        # --- DB Connection and Data Loading ---
        conn = sqlite3.connect(db_file)
        text_column = detect_text_column(conn)
        if not text_column:
             conn.close(); raise ValueError("No valid text column found.")
        
        year_column = detect_year_column(conn)
        select_columns = f"id AS paper_id, title, {text_column}"
        if year_column: select_columns += f", {year_column} AS year"
            
        query = f"SELECT {select_columns} FROM papers WHERE {text_column} IS NOT NULL AND {text_column} NOT LIKE 'Error%'"
        if year_column and year_range:
            query += f" AND {year_column} BETWEEN {year_range[0]} AND {year_range[1]}"
        df_papers = pd.read_sql_query(query, conn)
        conn.close()
        
        if df_papers.empty:
            update_log("No valid papers found for Seebeck extraction")
            return pd.DataFrame(columns=["paper_id", "title", "material", "seebeck_value", "seebeck_unit", "context"]), [], []

        # --- NLP Setup and Extraction ---
        nlp = load_spacy_model(st.session_state.synonyms)
        
        seebeck_extractions = []
        training_formulas = []
        training_targets = []
        total_papers = len(df_papers)

        for index, row in df_papers.iterrows():
            paper_id = row["paper_id"]
            title = row["title"]
            text = row[text_column]
            year = row["year"] if year_column else None
            
            update_progress(f"Processing paper {index + 1}/{total_papers}: {title}")

            doc = nlp(text)
            seebeck_mentions = find_seebeck_mentions(doc)
            linked_pairs = link_formula_to_seebeck(doc, seebeck_mentions)
            
            # Standardize and collect results
            for pair in linked_pairs:
                formula_text = pair["Formula_Span"].text
                normalized_formula = standardize_material_formula(formula_text, preserve_stoichiometry)
                
                if normalized_formula:
                    seebeck_value = pair["Seebeck_Value"]
                    seebeck_extractions.append({
                        "paper_id": paper_id,
                        "title": title,
                        "year": year,
                        "material": normalized_formula,
                        "seebeck_value": seebeck_value,
                        "seebeck_unit": pair["Seebeck_Unit"],
                        "context": pair["Context"]
                    })
                    # Collect data for GNN training
                    training_formulas.append(normalized_formula)
                    training_targets.append(seebeck_value)
                else:
                    update_log(f"Skipped formula standardization for paper {paper_id}")

        # --- Final Dataframe and Output ---
        results_df = pd.DataFrame(seebeck_extractions)
        
        update_log(f"Completed Seebeck extraction: Found {len(results_df)} valid entries.")
            
        return results_df, training_formulas, training_targets
        
    except Exception as e:
        update_log(f"Fatal error during Seebeck extraction: {str(e)}")
        # st.session_state.error_summary.append(f"Extraction failed: {str(e)}")
        return pd.DataFrame(columns=["paper_id", "title", "material", "seebeck_value", "seebeck_unit", "context"]), [], []

# Note: The Streamlit application entry point (which would call extract_seebeck_values,
# then optionally train_gnn_regressor, and then present predict_seebeck_gnn) is not
# included here but would use these functions.
