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
# Load spaCy model
# -----------------------------
def load_spacy_model(synonyms):
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except Exception as e:
        st.error(f"Failed to load spaCy: {e}. Install: `python -m spacy download en_core_web_sm`")
        st.stop()
    
    nlp.add_pipe("formula_ner", last=True)
    matcher = build_material_matcher(nlp, synonyms)
    nlp.add_pipe("material_matcher", last=True)
    
    if not Doc.has_extension("material_matcher"):
        Doc.set_extension("material_matcher", default=None)
    Doc.set_extension("material_matcher", default=matcher, force=True)
    
    if not Span.has_extension("norm"):
        Span.set_extension("norm", default=None)
    
    return nlp

# -----------------------------
# Link formulas to Seebeck value
# -----------------------------
def link_formula_to_seebeck(doc):
    formulas = [(ent, score_formula_context(ent.text, doc.text, st.session_state.synonyms)) 
                for ent in doc.ents if ent.label_ == "FORMULA"]
    formulas = [f for f, score in formulas if score > 0.3]
    seebeck_values = [ent for ent in doc.ents if ent.label_ == "SEEBECK_VALUE"]
    pairs = []
    for f in formulas:
        nearest_seebeck = None
        min_distance = float("inf")
        for s in seebeck_values:
            distance = abs(f.start_char - s.start_char)
            if distance < min_distance:
                min_distance = distance
                nearest_seebeck = s
        if nearest_seebeck:
            pairs.append({
                "Formula": f.text,
                "Seebeck_Value": nearest_seebeck.text,  # Would parse to float in extraction
            })
    return pairs

# -----------------------------
# Featurize formulas for GNN
# -----------------------------
def featurize_formulas(formulas, targets=None):
    """
    Convert formulas to graph data structures for GNN.
    Returns a list of PyTorch Geometric Data objects and valid formulas/targets.
    """
    data_list = []
    valid_formulas = []
    valid_targets = [] if targets is not None else None

    element_properties = {
        el.symbol: [
            float(el.Z or 0),
            float(el.X or 0),
            float(el.group or 0),
            float(el.row or 0),
            float(el.atomic_mass or 0)
        ] for el in Element
    }

    for i, formula in enumerate(formulas):
        if not validate_formula(formula):
            update_log(f"Skipped featurization for invalid formula '{formula}'")
            st.session_state.error_summary.append(f"Invalid formula '{formula}' for featurization")
            continue

        try:
            # Parse formula with pymatgen
            comp = Composition(formula)
            if not comp.valid:
                update_log(f"Invalid composition for formula '{formula}'")
                continue

            # Simplify fractional stoichiometries
            el_amt_dict = comp.get_el_amt_dict()
            el_amt_dict = {k: max(1, round(v)) for k, v in el_amt_dict.items()}  # Ensure at least 1 atom
            total_atoms = sum(el_amt_dict.values())
            if total_atoms < 2:
                update_log(f"Formula '{formula}' has fewer than 2 atoms: {el_amt_dict}")
                continue

            # Create a simple structure
            species = []
            frac_coords = []
            pos = 0
            for el, amt in el_amt_dict.items():
                for _ in range(int(amt)):
                    species.append(el)
                    frac_coords.append([pos * 0.1, 0, 0])  # Closer spacing to ensure edges
                    pos += 1

            if len(species) < 2:
                update_log(f"No valid species for formula '{formula}'")
                continue

            lattice = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
            structure = Structure(lattice, species, frac_coords, coords_are_cartesian=False)

            # Build structure graph with a robust strategy
            strategy = MinimumDistanceNN(cutoff=10.0)  # Increased cutoff
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
                # Fallback: fully connected graph
                update_log(f"No edges found for '{formula}'; using fully connected graph")
                for i in range(len(structure)):
                    for j in range(i + 1, len(structure)):
                        edge_index.append([i, j])
                        edge_index.append([j, i])  # Undirected
                        edge_weights.append(1.0)
            else:
                for i, neighbor_dict in enumerate(adjacency):
                    for neighbor_idx, data in neighbor_dict[1].items():
                        edge_index.append([i, neighbor_idx])
                        edge_weights.append(data.get('weight', 1.0))

            if not edge_index:
                update_log(f"No valid edges for formula '{formula}' after fallback")
                continue

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

            # Create Data object
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_weights.unsqueeze(-1),
                y=torch.tensor([targets[i]], dtype=torch.float) if targets is not None else None
            )

            data_list.append(data)
            valid_formulas.append(formula)
            if targets is not None:
                valid_targets.append(targets[i])

        except Exception as e:
            update_log(f"Failed to featurize formula '{formula}': {str(e)}")
            st.session_state.error_summary.append(f"Featurization failed for '{formula}': {str(e)}")
            continue

    if not data_list:
        update_log("No valid graph data generated for GNN")
        return [], [], [] if targets is not None else None

    update_log(f"Generated {len(data_list)} valid graph data objects")
    return data_list, valid_formulas, valid_targets if targets is not None else None

# -----------------------------
# GNN Model Definition for Regression
# -----------------------------
class GNNRegressor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1):
        super(GNNRegressor, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch_geometric.nn.global_mean_pool(x, data.batch)
        x = self.fc(x)
        return x.squeeze(-1)

# -----------------------------
# Train GNN Model
# -----------------------------
def train_gnn(formulas, targets):
    if not formulas or not targets:
        update_log("No valid data for GNN training")
        return None, None, {}

    # Featurize formulas into graph data
    data_list, valid_formulas, valid_targets = featurize_formulas(formulas, targets)
    if not data_list:
        update_log("No valid graph data for GNN training")
        return None, None, {}

    # Create DataLoader for batch processing
    dataset = data_list
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize GNN model
    model = GNNRegressor(input_dim=5, hidden_dim=64, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(100):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        update_log(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    # Placeholder scaler for compatibility
    scaler = StandardScaler()

    # Save models
    save_formats = st.session_state.get('save_formats', ["pkl", "db", "pt", "h5"])
    model_files = {}

    # SQLite Database (.db)
    if "db" in save_formats:
        try:
            conn = sqlite3.connect(st.session_state.db_file)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_type TEXT,
                    format TEXT,
                    model_data BLOB
                )
            """)
            temp_path = os.path.join(DB_DIR, "temp_gnn.pt")
            torch.save(model.state_dict(), temp_path)
            with open(temp_path, "rb") as f:
                model_data = f.read()
            cursor.execute(
                "INSERT INTO models (model_type, format, model_data) VALUES (?, ?, ?)",
                ("gnn_model", "pt", model_data)
            )
            conn.commit()
            conn.close()
            os.remove(temp_path)
            update_log("Saved GNN model to SQLite database")
        except Exception as e:
            update_log(f"Failed to save GNN model to SQLite database: {str(e)}")
            st.session_state.error_summary.append(f"SQLite save error: {str(e)}")

    # Pickle (.pkl)
    if "pkl" in save_formats:
        try:
            scaler_path = os.path.join(DB_DIR, "scaler.pkl")
            joblib.dump(scaler, scaler_path)
            model_files["scaler.pkl"] = scaler_path
            update_log(f"Saved scaler to {scaler_path}")
        except Exception as e:
            update_log(f"Failed to save .pkl scaler: {str(e)}")
            st.session_state.error_summary.append(f"Pickle save error: {str(e)}")

    # PyTorch (.pt)
    if "pt" in save_formats:
        try:
            model_path = os.path.join(DB_DIR, "gnn_model.pt")
            torch.save(model.state_dict(), model_path)
            model_files["gnn_model.pt"] = model_path
            update_log(f"Saved GNN model to {model_path}")

            scaler_params = {
                'mean': torch.tensor([0.0] * 5),
                'scale': torch.tensor([1.0] * 5)
            }
            scaler_path = os.path.join(DB_DIR, "scaler.pt")
            torch.save(scaler_params, scaler_path)
            model_files["scaler.pt"] = scaler_path
            update_log(f"Saved scaler to {scaler_path}")
        except Exception as e:
            update_log(f"Failed to save .pt files: {str(e)}")
            st.session_state.error_summary.append(f"PyTorch save error: {str(e)}")

    # HDF5 (.h5)
    if "h5" in save_formats:
        try:
            h5_path = os.path.join(DB_DIR, "gnn_models.h5")
            with h5py.File(h5_path, 'w') as f:
                model_group = f.create_group('gnn_model')
                for name, param in model.state_dict().items():
                    model_group.create_dataset(name, data=param.numpy())
                scaler_group = f.create_group('scaler')
                scaler_group.create_dataset('mean', data=np.zeros(5))
                scaler_group.create_dataset('scale', data=np.ones(5))
            model_files["gnn_models.h5"] = h5_path
            update_log(f"Saved GNN model to HDF5 file {h5_path}")
        except Exception as e:
            update_log(f"Failed to save .h5 file: {str(e)}")
            st.session_state.error_summary.append(f"HDF5 save error: {str(e)}")

    update_log(f"Trained GNN with {len(valid_formulas)} samples")
    return model, scaler, model_files

# -----------------------------
# Standardize material formula
# -----------------------------
def standardize_material_formula(formula, preserve_stoichiometry=False, canonical_order=True):
    if not formula or not isinstance(formula, str):
        update_log(f"Invalid input formula: {formula}")
        st.session_state.error_summary.append(f"Invalid formula: {formula}")
        return None
    
    formula = re.sub(r'\s+', '', formula)
    formula = re.sub(r'[\[\]\{\}]', '', formula)
    
    if not validate_formula(formula):
        update_log(f"Invalid formula '{formula}': failed validation")
        st.session_state.error_summary.append(f"Invalid formula '{formula}'")
        return None
    
    doping_pattern = r'(.+?)(?::|doped\s+)([A-Za-z0-9,\.]+)'
    doping_match = re.match(doping_pattern, formula, re.IGNORECASE)
    dopants = None
    if doping_match:
        base_formula, dopants = doping_match.groups()
        formula = base_formula.strip()
        dopants = dopants.split(',')
        update_log(f"Detected doped material: base='{formula}', dopants='{','.join(dopants)}'")
    
    try:
        comp = Composition(formula)
        if not comp.valid:
            update_log(f"Invalid chemical formula '{formula}': not a valid composition")
            st.session_state.error_summary.append(f"Invalid formula '{formula}': not a valid composition")
            return None
        
        elements = comp.elements
        if not all(isinstance(el, Element) for el in elements):
            update_log(f"Invalid elements in formula '{formula}'")
            st.session_state.error_summary.append(f"Invalid elements in formula '{formula}'")
            return None
        
        if preserve_stoichiometry:
            el_amt_dict = comp.get_el_amt_dict()
            standardized_formula = ''.join(
                f"{el}{amt:.2f}" if amt != int(amt) else f"{el}{int(amt)}"
                for el, amt in (sorted(el_amt_dict.items()) if canonical_order else el_amt_dict.items())
            )
        else:
            standardized_formula = comp.reduced_formula
        
        if dopants:
            valid_dopants = []
            for dopant in dopants:
                if not validate_formula(dopant):
                    update_log(f"Invalid dopant '{dopant}' in '{formula}'")
                    st.session_state.error_summary.append(f"Invalid dopant '{dopant}' in '{formula}'")
                    continue
                try:
                    dopant_comp = Composition(dopant.strip())
                    valid_dopants.append(dopant_comp.reduced_formula)
                except Exception as e:
                    update_log(f"Failed to parse dopant '{dopant}' in '{formula}': {e}")
                    st.session_state.error_summary.append(f"Failed to parse dopant '{dopant}' in '{formula}'")
            if valid_dopants:
                standardized_formula = f"{standardized_formula}:{','.join(valid_dopants)}"
        
        update_log(f"Standardized formula '{formula}' to '{standardized_formula}' using pymatgen")
        return standardized_formula
    except Exception as e:
        update_log(f"pymatgen could not parse formula '{formula}': {str(e)}")
        st.session_state.error_summary.append(f"pymatgen failed for '{formula}': {str(e)}")
        return None

# -----------------------------
# Predict Seebeck using GNN
# -----------------------------
def predict_seebeck(formula, material_df, fuzzy_match=False):
    try:
        if not formula.strip():
            update_log("Empty formula input provided")
            return None, "Please enter a valid chemical formula.", None
        
        normalized_formula = standardize_material_formula(formula, 
                                                        preserve_stoichiometry=st.session_state.get('preserve_stoichiometry', False))
        if not normalized_formula:
            update_log(f"Invalid chemical formula: {formula}")
            return None, f"'{formula}' is not a valid chemical formula.", None
        
        update_log(f"Normalized formula '{formula}' to '{normalized_formula}'")
        
        if material_df is None or material_df.empty:
            update_log("No Seebeck data available for formula lookup")
            return None, "Please run Seebeck Analysis first.", None
        
        formula_matches = material_df[material_df["material"].str.lower() == normalized_formula.lower()]
        similar_formula = None
        
        if formula_matches.empty and fuzzy_match:
            materials = material_df["material"].unique()
            similarities = [(m, SequenceMatcher(None, normalized_formula.lower(), m.lower()).ratio()) for m in materials]
            best_match, similarity = max(similarities, key=lambda x: x[1]) if similarities else (None, 0)
            if similarity > 0.8:
                formula_matches = material_df[material_df["material"].str.lower() == best_match.lower()]
                similar_formula = best_match
                update_log(f"Fuzzy matched '{normalized_formula}' to '{best_match}' (similarity: {similarity:.2%})")
        
        if not formula_matches.empty:
            seebeck_values = formula_matches["seebeck"].tolist()
            avg_seebeck = np.mean(seebeck_values)
            std_seebeck = np.std(seebeck_values)
            total_matches = len(formula_matches)
            paper_ids = formula_matches["paper_id"].unique()
            contexts = formula_matches["context"].tolist()
            
            update_log(f"Formula '{normalized_formula}' has average Seebeck {avg_seebeck:.2f} μV/K (std: {std_seebeck:.2f})")
            return {
                "formula": normalized_formula,
                "seebeck": avg_seebeck,
                "std": std_seebeck,
                "paper_ids": paper_ids.tolist(),
                "count": total_matches,
                "contexts": contexts,
                "all_values": seebeck_values
            }, None, similar_formula
        else:
            if st.session_state.ann_model is None:
                update_log("No GNN model available for prediction")
                return None, "Please run Seebeck Analysis to train the GNN.", None
            
            # Featurize the single formula
            data_list, valid_formulas, _ = featurize_formulas([normalized_formula])
            if not data_list:
                update_log(f"Failed to featurize formula '{normalized_formula}' for GNN")
                return None, f"Could not featurize formula '{normalized_formula}' for prediction.", None
            
            # Check if .pt model is available
            if "gnn_model.pt" in st.session_state.model_files:
                try:
                    model = GNNRegressor(input_dim=5, hidden_dim=64, output_dim=1)
                    model.load_state_dict(torch.load(st.session_state.model_files["gnn_model.pt"], map_location='cpu'))
                    model.eval()

                    data = data_list[0]
                    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)  # Single graph batch
                    with torch.no_grad():
                        prediction = model(data).item()

                    update_log(f"GNN predicted '{normalized_formula}' Seebeck {prediction:.2f} μV/K")
                    return {
                        "formula": normalized_formula,
                        "seebeck": prediction,
                        "std": 0.0,
                        "paper_ids": [],
                        "count": 0,
                        "contexts": [],
                        "all_values": [prediction]
                    }, None, None
                except Exception as e:
                    update_log(f"GNN prediction failed: {str(e)}")
                    st.session_state.error_summary.append(f"GNN prediction error: {str(e)}")
            
            update_log(f"No GNN model (.pt) found for prediction")
            return None, "No GNN model available for prediction.", None
    
    except Exception as e:
        update_log(f"Error predicting Seebeck for '{formula}': {str(e)}")
        return None, f"Error predicting Seebeck: {str(e)}", None

# -----------------------------
# Batch predict Seebeck
# -----------------------------
def batch_predict_seebeck(formulas, material_df, fuzzy_match=False):
    results = []
    errors = []
    suggestions = []
    for formula in formulas:
        result, error, similar_formula = predict_seebeck(formula.strip(), material_df, fuzzy_match)
        if error:
            errors.append(error)
            if similar_formula:
                suggestions.append((formula, similar_formula))
        else:
            results.append(result)
    return results, errors, suggestions

# -----------------------------
# Extract Seebeck values
# -----------------------------
def extract_seebeck_values(db_file, preserve_stoichiometry=False, year_range=None):
    try:
        update_log("Starting Seebeck value extraction with NER")
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
            update_log("No valid papers found for Seebeck extraction")
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
        
        progress_bar = st.progress(0)
        for i, row in df.iterrows():
            update_progress(f"Processing paper {row['paper_id']} ({i+1}/{len(df)})")
            content = row[text_column]
            chunks = chunk_text(content)
            
            for chunk_idx, chunk in enumerate(chunks):
                doc = nlp(chunk)
                formula_entities = [ent.text for ent in doc.ents if ent.label_ == "FORMULA"]
                material_entities = [ent for ent in doc.ents if ent.label_ == "MATERIAL_TYPE"]
                
                linked_pairs = link_formula_to_seebeck(doc)
                
                for pair in linked_pairs:
                    extraction_entry = {
                        "paper_id": row["paper_id"],
                        "title": row["title"],
                        "material": pair["Formula"],
                        "seebeck": pair["Seebeck_Value"],
                        "context": f"Found in context: {chunk[max(0, chunk.find(pair['Formula'])-50):min(len(chunk), chunk.find(pair['Formula'])+50)]}..."
                    }
                    if 'year' in row:
                        extraction_entry['year'] = row['year']
                    seebeck_extractions.append(extraction_entry)
                
                seebeck_materials = set()
                for pattern in seebeck_patterns:
                    matches = re.finditer(pattern, chunk, re.IGNORECASE)
                    for match in matches:
                        value_str = match.group(1).strip()
                        try:
                            value = float(value_str)
                            if -500 <= value <= 500:
                                material = match.group(2).strip() if len(match.groups()) > 1 else ""
                                if material and len(material) > 2 and material in formula_entities and validate_formula(material):
                                    standardized_material = standardize_material_formula(material, preserve_stoichiometry)
                                    if standardized_material:
                                        seebeck_materials.add((standardized_material, value, match.start()))
                        except ValueError:
                            continue
                
                seebeck_context = re.search(r"seebeck[^\.]{0,500}", chunk, re.IGNORECASE)
                
                if seebeck_context:
                    context_doc = nlp(seebeck_context.group(0))
                    for ent in context_doc.ents:
                        if ent.label_ == "FORMULA" and validate_formula(ent.text):
                            standardized_material = standardize_material_formula(ent.text, preserve_stoichiometry)
                            if standardized_material:
                                # Extract value from context (simplified; in practice, parse nearby number)
                                value_match = re.search(r'([-+]?\d+(?:\.\d+)?)', seebeck_context.group(0))
                                if value_match:
                                    value = float(value_match.group(1))
                                    if -500 <= value <= 500:
                                        seebeck_materials.add((standardized_material, value, ent.start_char))
                
                for material in common_te_materials:
                    if material.lower() in chunk.lower():
                        doc = nlp(material)
                        if any(ent.label_ == "FORMULA" for ent in doc.ents) and validate_formula(material):
                            standardized_material = standardize_material_formula(material, preserve_stoichiometry)
                            if standardized_material:
                                if seebeck_context and material.lower() in seebeck_context.group(0).lower():
                                    value_match = re.search(r'([-+]?\d+(?:\.\d+)?)', seebeck_context.group(0))
                                    if value_match:
                                        value = float(value_match.group(1))
                                        if -500 <= value <= 500:
                                            seebeck_materials.add((standardized_material, value, 0))
                
                for material, value, start_pos in seebeck_materials:
                    context = chunk[max(0, start_pos-50):min(len(chunk), start_pos+50)]
                    extraction_entry = {
                        "paper_id": row["paper_id"],
                        "title": row["title"],
                        "material": material,
                        "seebeck": value,
                        "context": f"Found in context: {context}..."
                    }
                    if 'year' in row:
                        extraction_entry['year'] = row['year']
                    seebeck_extractions.append(extraction_entry)
                
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
        update_log(f"Cleaned and sorted DataFrame: {len(seebeck_df)} unique extractions")
        update_log(f"seebeck_df columns: {seebeck_df.columns.tolist()}")
        
        conn = sqlite3.connect(db_file)
        seebeck_df[["material", "seebeck"] + (["year"] if 'year' in seebeck_df.columns else [])].to_sql("standardized_formulas", conn, if_exists="replace", index=False)
        conn.close()
        update_log("Cached standardized formulas in database")
        
        formulas = seebeck_df["material"].unique().tolist()
        targets = [seebeck_df[seebeck_df["material"] == f]["seebeck"].mean() for f in formulas]
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
# Plot Seebeck values
# -----------------------------
def plot_seebeck_values(df, top_n=20, year_range=None):
    if df.empty:
        update_log("Empty DataFrame provided to plot_seebeck_values")
        return None, None, None, None, None
    
    update_log(f"DataFrame columns: {df.columns.tolist()}")
    
    # Apply year range filter if 'year' column exists
    if year_range and 'year' in df.columns:
        try:
            df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
            update_log(f"Filtered DataFrame by year range {year_range}: {len(df)} rows")
        except Exception as e:
            update_log(f"Error filtering by year: {str(e)}")
            st.session_state.error_summary.append(f"Year filter error: {str(e)}")
            df = df.copy()
    elif year_range and 'year' not in df.columns:
        update_log("Year column not found in DataFrame; skipping year filter")
        st.session_state.error_summary.append("Year column not found; visualizations will exclude year-based filtering")
    
    if df.empty:
        update_log("No data after filtering")
        return None, None, None, None, None
    
    material_averages = df.groupby("material")["seebeck"].agg(['mean', 'count']).reset_index()
    top_materials = material_averages.sort_values("mean", ascending=False)["material"].head(top_n).tolist()
    filtered_df = df[df["material"].isin(top_materials)]
    
    # Bar chart
    fig_bar = px.bar(
        material_averages.sort_values("mean", ascending=False).head(top_n), 
        x="material", 
        y="mean", 
        error_y="std" if 'std' in material_averages else None,
        title=f"Top {top_n} Materials by Average Seebeck (μV/K)",
        labels={"material": "Formula", "mean": "Average Seebeck (μV/K)"},
        color_discrete_sequence=["#636EFA"]
    )
    fig_bar.update_layout(xaxis_tickangle=-45, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    
    # Histogram
    fig_hist = px.histogram(
        filtered_df, x="seebeck",
        title="Distribution of Seebeck Values",
        labels={"seebeck": "Seebeck (μV/K)"},
        color_discrete_sequence=["#636EFA"]
    )
    fig_hist.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    
    # Timeline chart
    fig_timeline = None
    if 'year' in df.columns and df["year"].notna().any():
        yearly_data = df.groupby("year")["seebeck"].mean().reset_index()
        fig_timeline = px.line(
            yearly_data,
            x="year",
            y="seebeck",
            title="Trend of Average Seebeck Over Time",
            labels={"year": "Year", "seebeck": "Average Seebeck (μV/K)"},
            color_discrete_sequence=["#636EFA"]
        )
        fig_timeline.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    else:
        update_log("No valid year data for timeline plot")
    
    # Co-occurrence heatmap
    material_papers = df.groupby(["material", "paper_id"]).size().unstack(fill_value=0)
    co_occurrence = material_papers.T.dot(material_papers)
    np.fill_diagonal(co_occurrence.values, 0)
    
    valid_materials = [m for m in top_materials if m in co_occurrence.index and m in co_occurrence.columns]
    update_log(f"Top materials: {list(top_materials)}")
    update_log(f"Valid materials for co-occurrence: {valid_materials}")
    update_log(f"Co-occurrence index: {list(co_occurrence.index)}")
    
    if not valid_materials:
        update_log("No valid materials for co-occurrence heatmap")
        fig_heatmap = None
    else:
        co_occurrence = co_occurrence.loc[valid_materials, valid_materials]
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=co_occurrence.values,
            x=co_occurrence.columns,
            y=co_occurrence.index,
            colorscale="Viridis",
            text=co_occurrence.values,
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        fig_heatmap.update_layout(
            title="Material Co-occurrence Heatmap",
            xaxis_title="Formula",
            yaxis_title="Formula",
            xaxis_tickangle=-45,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
    
    # Sunburst chart
    fig_sunburst = None
    if 'year' in df.columns:
        sunburst_data = df.groupby(['year', 'material']).agg({'seebeck': 'mean'}).reset_index()
        fig_sunburst = px.sunburst(
            sunburst_data,
            path=['year', 'material'],
            values='seebeck',
            title="Hierarchical Distribution of Seebeck Values",
            labels={"year": "Year", "material": "Formula", "seebeck": "Average Seebeck (μV/K)"}
        )
        fig_sunburst.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    else:
        sunburst_data = df.groupby(['material']).agg({'seebeck': 'mean'}).reset_index()
        fig_sunburst = px.sunburst(
            sunburst_data,
            path=['material'],
            values='seebeck',
            title="Hierarchical Distribution of Seebeck Values (No Year Data)",
            labels={"material": "Formula", "seebeck": "Average Seebeck (μV/K)"}
        )
        fig_sunburst.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    
    return fig_bar, fig_hist, fig_timeline, fig_heatmap, fig_sunburst

# -----------------------------
# Logging and directory setup
# -----------------------------
DB_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(
    filename=os.path.join(DB_DIR, 'thermoelectric_seebeck_analysis.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ... (rest of utils: update_log, update_progress, detect_text_column, detect_year_column remain identical)

# -----------------------------
# Main Streamlit app
# -----------------------------
# ... (identical to original, but replace "Material Classification" with "Seebeck Value Extraction", "p-type vs n-type" with "Seebeck coefficients", synonyms for Seebeck terms, call extract_seebeck_values, plot_seebeck_values, predict_seebeck, batch_predict_seebeck, update metrics/visuals for numerical Seebeck, CSV "seebeck_via_nlp.csv", etc.)
else:
    st.warning("Select or upload a database file.")
