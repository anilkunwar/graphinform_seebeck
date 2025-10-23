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
# Seebeck value extraction patterns
# -----------------------------
def extract_seebeck_value(text):
    """Extract Seebeck values from text with comprehensive patterns."""
    patterns = [
        r"seebeck\s+coefficient[s]?\s*[=:]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:μV/K|µV/K|μV·K⁻¹|µV·K⁻¹)",
        r"s\s*[=:]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:μV/K|µV/K|μV·K⁻¹|µV·K⁻¹)",
        r"α\s*[=:]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:μV/K|µV/K|μV·K⁻¹|µV·K⁻¹)",
        r"thermoelectric\s+power\s*[=:]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:μV/K|µV/K|μV·K⁻¹|µV·K⁻¹)",
        r"seebeck\s*[=:]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:μV/K|µV/K|μV·K⁻¹|µV·K⁻¹)",
        r"([-+]?\d+(?:\.\d+)?)\s*(?:μV/K|µV/K|μV·K⁻¹|µV·K⁻¹)\s*(?:for|of|in)\s+[A-Za-z]",
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                value = float(match.group(1))
                if -500 <= value <= 500:  # Reasonable Seebeck range
                    return value
            except ValueError:
                continue
    return None

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
            el_amt_dict = {k: max(1, round(v)) for k, v in el_amt_dict.items()}
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
                    frac_coords.append([pos * 0.1, 0, 0])
                    pos += 1

            if len(species) < 2:
                update_log(f"No valid species for formula '{formula}'")
                continue

            lattice = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
            structure = Structure(lattice, species, frac_coords, coords_are_cartesian=False)

            # Build structure graph
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
                update_log(f"No edges found for '{formula}'; using fully connected graph")
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

    data_list, valid_formulas, valid_targets = featurize_formulas(formulas, targets)
    if not data_list:
        update_log("No valid graph data for GNN training")
        return None, None, {}

    dataset = data_list
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GNNRegressor(input_dim=5, hidden_dim=64, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

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

    scaler = StandardScaler()
    save_formats = st.session_state.get('save_formats', ["pkl", "db", "pt", "h5"])
    model_files = {}

    # Save in multiple formats
    if "pt" in save_formats:
        try:
            model_path = os.path.join(DB_DIR, "gnn_regressor.pt")
            torch.save(model.state_dict(), model_path)
            model_files["gnn_regressor.pt"] = model_path
            update_log(f"Saved GNN regressor to {model_path}")
        except Exception as e:
            update_log(f"Failed to save .pt model: {str(e)}")

    update_log(f"Trained GNN regressor with {len(valid_formulas)} samples")
    return model, scaler, model_files

# -----------------------------
# Standardize material formula
# -----------------------------
def standardize_material_formula(formula, preserve_stoichiometry=False):
    if not formula or not isinstance(formula, str):
        return None
    
    formula = re.sub(r'\s+', '', formula)
    formula = re.sub(r'[\[\]\{\}]', '', formula)
    
    if not validate_formula(formula):
        return None
    
    doping_pattern = r'(.+?)(?::|doped\s+)([A-Za-z0-9,\.]+)'
    doping_match = re.match(doping_pattern, formula, re.IGNORECASE)
    dopants = None
    if doping_match:
        base_formula, dopants_str = doping_match.groups()
        formula = base_formula.strip()
        dopants = [d.strip() for d in dopants_str.split(',')]
    
    try:
        comp = Composition(formula)
        if not comp.valid:
            return None
        
        standardized_formula = comp.reduced_formula
        
        if dopants:
            valid_dopants = []
            for dopant in dopants:
                try:
                    dopant_comp = Composition(dopant)
                    valid_dopants.append(dopant_comp.reduced_formula)
                except:
                    continue
            if valid_dopants:
                standardized_formula = f"{standardized_formula}:{','.join(valid_dopants)}"
        
        return standardized_formula
    except Exception:
        return None

# -----------------------------
# Predict Seebeck using GNN
# -----------------------------
def predict_seebeck(formula, material_df, fuzzy_match=False):
    try:
        if not formula.strip():
            return None, "Please enter a valid chemical formula.", None
        
        normalized_formula = standardize_material_formula(formula, 
                                                        st.session_state.get('preserve_stoichiometry', False))
        if not normalized_formula:
            return None, f"'{formula}' is not a valid chemical formula.", None
        
        if material_df is None or material_df.empty:
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
        
        if not formula_matches.empty:
            seebeck_values = formula_matches["seebeck"].tolist()
            avg_seebeck = np.mean(seebeck_values)
            std_seebeck = np.std(seebeck_values)
            return {
                "formula": normalized_formula,
                "seebeck": avg_seebeck,
                "std": std_seebeck,
                "count": len(formula_matches),
                "all_values": seebeck_values
            }, None, similar_formula
        else:
            if st.session_state.ann_model is None:
                return None, "Please run Seebeck Analysis to train the GNN.", None
            
            data_list, valid_formulas, _ = featurize_formulas([normalized_formula])
            if not data_list:
                return None, f"Could not featurize formula '{normalized_formula}'.", None
            
            try:
                model = GNNRegressor(input_dim=5, hidden_dim=64, output_dim=1)
                model.load_state_dict(torch.load(st.session_state.model_files["gnn_regressor.pt"], map_location='cpu'))
                model.eval()

                data = data_list[0]
                data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
                with torch.no_grad():
                    prediction = model(data).item()

                return {
                    "formula": normalized_formula,
                    "seebeck": prediction,
                    "std": 0.0,
                    "count": 0,
                    "all_values": [prediction],
                    "source": "GNN Prediction"
                }, None, None
            except Exception as e:
                return None, f"GNN prediction failed: {str(e)}", None
    
    except Exception as e:
        return None, f"Error predicting Seebeck: {str(e)}", None

# -----------------------------
# Extract Seebeck values
# -----------------------------
def extract_seebeck_values(db_file, preserve_stoichiometry=False, year_range=None):
    try:
        update_log("Starting Seebeck coefficient extraction")
        
        conn = sqlite3.connect(db_file)
        text_column = detect_text_column(conn)
        if not text_column:
            conn.close()
            return pd.DataFrame()
        
        year_column = detect_year_column(conn)
        select_columns = f"id AS paper_id, title, {text_column}"
        if year_column:
            select_columns += f", {year_column} AS year"
        
        query = f"SELECT {select_columns} FROM papers WHERE {text_column} IS NOT NULL AND {text_column} NOT LIKE 'Error%'"
        if year_column and year_range:
            query += f" AND {year_column} BETWEEN {year_range[0]} AND {year_range[1]}"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return pd.DataFrame()
        
        nlp = load_spacy_model(st.session_state.synonyms)
        seebeck_extractions = []
        
        progress_bar = st.progress(0)
        for i, row in df.iterrows():
            content = row[text_column]
            
            # Extract Seebeck values and associate with formulas
            seebeck_value = extract_seebeck_value(content)
            if seebeck_value is not None:
                doc = nlp(content)
                for ent in doc.ents:
                    if ent.label_ == "FORMULA" and validate_formula(ent.text):
                        standardized_formula = standardize_material_formula(ent.text, preserve_stoichiometry)
                        if standardized_formula:
                            seebeck_extractions.append({
                                "paper_id": row["paper_id"],
                                "title": row["title"],
                                "material": standardized_formula,
                                "seebeck": seebeck_value,
                                "context": content[max(0, ent.start_char-50):min(len(content), ent.end_char+50)]
                            })
            
            progress_value = min((i + 1) / len(df), 1.0)
            progress_bar.progress(progress_value)
        
        seebeck_df = pd.DataFrame(seebeck_extractions)
        if seebeck_df.empty:
            return pd.DataFrame()
        
        # Cache to database
        conn = sqlite3.connect(db_file)
        seebeck_df[["material", "seebeck"]].to_sql("seebeck_cache", conn, if_exists="replace", index=False)
        conn.close()
        
        # Train GNN on averaged values per formula
        material_averages = seebeck_df.groupby("material")["seebeck"].mean().to_dict()
        formulas = list(material_averages.keys())
        targets = list(material_averages.values())
        
        model, scaler, model_files = train_gnn(formulas, targets)
        st.session_state.ann_model = model
        st.session_state.model_files = model_files
        
        return seebeck_df
    
    except Exception as e:
        update_log(f"Error in Seebeck extraction: {str(e)}")
        return pd.DataFrame()

# -----------------------------
# Plot Seebeck values
# -----------------------------
def plot_seebeck_values(df, top_n=20):
    if df.empty:
        return None
    
    material_stats = df.groupby("material")["seebeck"].agg(['mean', 'std', 'count']).reset_index()
    top_materials = material_stats.nlargest(top_n, 'mean')
    
    # Bar chart
    fig_bar = px.bar(
        top_materials, 
        x="material", 
        y="mean",
        error_y="std",
        title=f"Top {top_n} Materials by Average Seebeck Coefficient (μV/K)",
        labels={"material": "Formula", "mean": "Average Seebeck (μV/K)"}
    )
    fig_bar.update_layout(xaxis_tickangle=-45)
    
    # Histogram
    fig_hist = px.histogram(df, x="seebeck", title="Distribution of Seebeck Coefficients", nbins=50)
    
    return fig_bar, fig_hist

# -----------------------------
# Logging functions
# -----------------------------
logging.basicConfig(
    filename=os.path.join(DB_DIR, 'seebeck_gnn_analysis.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def update_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    memory_usage = psutil.Process().memory_info().rss / 1024**2
    log_message = f"[{timestamp}] {message} (Memory: {memory_usage:.2f} MB)"
    
    if "log_buffer" not in st.session_state:
        st.session_state.log_buffer = []
    st.session_state.log_buffer.append(log_message)
    if len(st.session_state.log_buffer) > 50:
        st.session_state.log_buffer.pop(0)
    
    logging.info(log_message)

def update_progress(message):
    if "progress_log" not in st.session_state:
        st.session_state.progress_log = []
    st.session_state.progress_log.append(message)
    if len(st.session_state.progress_log) > 10:
        st.session_state.progress_log.pop(0)

def detect_text_column(conn):
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(papers)")
    columns = {col[1].lower() for col in cursor.fetchall()}
    possible_text_columns = ['content', 'text', 'abstract', 'body']
    for col in possible_text_columns:
        if col.lower() in columns:
            return col
    return None

def detect_year_column(conn):
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(papers)")
    columns = {col[1].lower() for col in cursor.fetchall()}
    possible_year_columns = ['year', 'publication_year', 'date']
    for col in possible_year_columns:
        if col.lower() in columns:
            return col
    return None

# -----------------------------
# Main Streamlit App
# -----------------------------
st.set_page_config(page_title="Seebeck GNN Regressor", layout="wide")
st.title("🔬 Seebeck Coefficient GNN Regressor")
st.markdown("""
**Extract and predict Seebeck coefficients (μV/K) from thermoelectric literature using GNN**
- NLP extraction of formulas + Seebeck values from SQLite databases
- Graph Neural Network trained on extracted data
- Predict Seebeck for any chemical formula
""")

# Initialize session state
for key in ["log_buffer", "seebeck_data", "db_file", "error_summary", "progress_log", 
            "text_column", "synonyms", "ann_model", "model_files", "save_formats"]:
    if key not in st.session_state:
        if key == "synonyms":
            st.session_state[key] = {
                "seebeck": ["seebeck coefficient", "seebeck", "thermopower", "s", "α"],
                "material": ["thermoelectric material", "te material", "semiconductor"]
            }
        elif key == "save_formats":
            st.session_state[key] = ["pt", "pkl"]
        elif key == "log_buffer":
            st.session_state[key] = []
        elif key == "error_summary":
            st.session_state[key] = []
        elif key == "progress_log":
            st.session_state[key] = []
        else:
            st.session_state[key] = None

# Database selection
st.header("📁 Database Selection")
db_files = glob.glob(os.path.join(DB_DIR, "*.db"))
db_options = [os.path.basename(f) for f in db_files] + ["Upload new .db"]
db_selection = st.selectbox("Select Database", db_options)

if db_selection != "Upload new .db":
    st.session_state.db_file = os.path.join(DB_DIR, db_selection)
    update_log(f"Selected database: {db_selection}")
else:
    uploaded_file = st.file_uploader("Upload SQLite (.db)", type=["db"])
    if uploaded_file:
        temp_path = os.path.join(DB_DIR, f"uploaded_{uuid.uuid4().hex}.db")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.db_file = temp_path
        update_log(f"Uploaded database: {temp_path}")

# Main tabs
if st.session_state.db_file:
    tab1, tab2 = st.tabs(["📊 Seebeck Extraction", "🔮 Formula Prediction"])
    
    with tab1:
        st.header("Seebeck Coefficient Extraction & Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            preserve_stoichiometry = st.checkbox("Preserve stoichiometry", value=False)
            top_n = st.slider("Top N materials", 5, 30, 15)
        with col2:
            st.session_state.save_formats = st.multiselect(
                "Save model formats", ["pt", "pkl", "db", "h5"], 
                default=st.session_state.save_formats
            )
        
        if st.button("🚀 Extract Seebeck Coefficients", type="primary"):
            with st.spinner("Extracting Seebeck values..."):
                st.session_state.seebeck_data = extract_seebeck_values(
                    st.session_state.db_file, preserve_stoichiometry
                )
            
            if not st.session_state.seebeck_data.empty:
                st.success(f"✅ Extracted {len(st.session_state.seebeck_data)} Seebeck values!")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Total Extractions", len(st.session_state.seebeck_data))
                with col2: st.metric("Avg Seebeck", f"{st.session_state.seebeck_data['seebeck'].mean():.1f} μV/K")
                with col3: st.metric("Max Seebeck", f"{st.session_state.seebeck_data['seebeck'].max():.1f} μV/K")
                with col4: st.metric("Unique Materials", st.session_state.seebeck_data["material"].nunique())
                
                # Visualizations
                fig_bar, fig_hist = plot_seebeck_values(st.session_state.seebeck_data, top_n)
                col1, col2 = st.columns(2)
                with col1: st.plotly_chart(fig_bar, use_container_width=True)
                with col2: st.plotly_chart(fig_hist, use_container_width=True)
                
                # Data table
                st.subheader("Extracted Data")
                st.dataframe(st.session_state.seebeck_data.head(100))
                
                # Download
                csv = st.session_state.seebeck_data.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, "seebeck_extractions.csv", "text/csv")
            else:
                st.warning("No Seebeck values extracted. Check logs.")
        
        st.subheader("📋 Logs")
        st.text_area("", "\n".join(st.session_state.log_buffer), height=200)
    
    with tab2:
        st.header("🔮 Predict Seebeck Coefficient")
        
        col1, col2 = st.columns(2)
        with col1:
            formula_input = st.text_input("Enter Chemical Formula", placeholder="e.g., Bi2Te3, PbTe, SnSe")
            fuzzy_match = st.checkbox("Enable fuzzy matching")
        with col2:
            if st.button("🔮 Predict Seebeck", type="primary"):
                if formula_input:
                    with st.spinner("Predicting..."):
                        result, error, similar = predict_seebeck(
                            formula_input, st.session_state.seebeck_data, fuzzy_match
                        )
                        
                        if error:
                            st.error(error)
                            if similar:
                                st.info(f"💡 Similar formula found: {similar}")
                        else:
                            st.success(f"✅ **{result['formula']}**")
                            st.metric("Predicted Seebeck", f"{result['seebeck']:.1f} μV/K", 
                                    delta=f"±{result['std']:.1f}" if result['std'] > 0 else None)
                            
                            if result['count'] > 0:
                                st.info(f"📚 Found in {result['count']} papers")
                            else:
                                st.info("🧠 GNN prediction")
                            
                            st.json({"All values": result['all_values'][:5]})
        
        st.subheader("Batch Prediction")
        uploaded_csv = st.file_uploader("Upload CSV with 'formula' column")
        if uploaded_csv and st.button("Predict Batch"):
            df = pd.read_csv(uploaded_csv)
            if 'formula' in df.columns:
                results = []
                for formula in df['formula'].dropna():
                    result, _, _ = predict_seebeck(formula, st.session_state.seebeck_data, fuzzy_match)
                    if result:
                        results.append(result)
                
                if results:
                    batch_df = pd.DataFrame([{
                        "Formula": r["formula"],
                        "Predicted Seebeck (μV/K)": f"{r['seebeck']:.1f}",
                        "Source": "Literature" if r['count'] > 0 else "GNN"
                    } for r in results])
                    st.dataframe(batch_df)
                    st.download_button("Download Batch Results", batch_df.to_csv(index=False), "batch_predictions.csv")
            else:
                st.error("CSV must have 'formula' column")

else:
    st.warning("👆 Please select or upload a database file first.")

# Model downloads
if st.session_state.model_files:
    st.sidebar.header("💾 Download Models")
    for filename, filepath in st.session_state.model_files.items():
        with open(filepath, 'rb') as f:
            st.sidebar.download_button(f"Download {filename}", f, file_name=filename)
