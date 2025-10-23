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

# Seebeck coefficient extraction patterns
SEEBECK_PATTERNS = {
    # S = value μV/K
    r'S\s*=\s*([-+]?\d+(?:\.\d+)?)\s*μV/K': lambda m: float(m.group(1)),
    r'S\s*=\s*([-+]?\d+(?:\.\d+)?)\s*μV·K⁻¹': lambda m: float(m.group(1)),
    r'Seebeck\s*=\s*([-+]?\d+(?:\.\d+)?)\s*μV/K': lambda m: float(m.group(1)),
    r'Thermoelectric\s*power\s*=\s*([-+]?\d+(?:\.\d+)?)\s*μV/K': lambda m: float(m.group(1)),
    
    # α = value μV/K
    r'α\s*=\s*([-+]?\d+(?:\.\d+)?)\s*μV/K': lambda m: float(m.group(1)),
    r'α\s*=\s*([-+]?\d+(?:\.\d+)?)\s*μV·K⁻¹': lambda m: float(m.group(1)),
    
    # Various unit formats
    r'S\s*=\s*([-+]?\d+(?:\.\d+)?)\s*×?\s*10⁻⁶\s*V/K': lambda m: float(m.group(1)),  # μV/K
    r'S\s*=\s*([-+]?\d+(?:\.\d+)?)\s*mV/K': lambda m: float(m.group(1)) * 1000,     # Convert mV/K to μV/K
    r'S\s*=\s*([-+]?\d+(?:\.\d+)?)\s*μV/K': lambda m: float(m.group(1)),
    
    # With ± error
    r'S\s*=\s*([-+]?\d+(?:\.\d+)?)\s*(?:±\s*\d+(?:\.\d+)?)?\s*μV/K': lambda m: float(m.group(1)),
    r'S\s*=\s*([-+]?\d+(?:\.\d+)?)\s*(?:±\s*\d+(?:\.\d+)?)?\s*μV·K⁻¹': lambda m: float(m.group(1)),
    
    # Table-like patterns
    r'([-+]?\d+(?:\.\d+)?)\s*μV/K': lambda m: float(m.group(1)),
    r'([-+]?\d+(?:\.\d+)?)\s*μV·K⁻¹': lambda m: float(m.group(1)),
}

# Unit conversion factors to μV/K
UNIT_CONVERSIONS = {
    'μV/K': 1.0,
    'μV·K⁻¹': 1.0,
    'mV/K': 1000.0,
    'V/K': 1e6,
}

# -----------------------------
# Regex NER for formulas (unchanged)
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
    
    if len(base_formula) <= 2:
        return False
    
    try:
        comp = Composition(base_formula)
        if not comp.valid:
            return False
        elements = [el.symbol for el in comp.elements]
        total_atoms = sum(comp.get_el_amt_dict().values())
        if total_atoms < 2:
            return False
        return all(el in VALID_ELEMENTS for el in elements)
    except Exception:
        return False

# -----------------------------
# Enhanced Seebeck extraction
# -----------------------------
def extract_seebeck_from_text(text, formulas):
    """Extract Seebeck coefficients and link to nearest formulas."""
    seebeck_extractions = []
    
    for pattern, extractor in SEEBECK_PATTERNS.items():
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for match in matches:
            seebeck_value = extractor(match)
            if -500 <= seebeck_value <= 500:  # Reasonable range
                # Find nearest formula within context window
                start_pos = match.start()
                context_start = max(0, start_pos - 200)
                context_end = min(len(text), start_pos + 200)
                context = text[context_start:context_end]
                
                nearest_formula = None
                min_distance = float('inf')
                
                for formula in formulas:
                    formula_pos = text.lower().find(formula.lower())
                    if context_start <= formula_pos <= context_end:
                        distance = abs(formula_pos - start_pos)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_formula = formula
                
                if nearest_formula:
                    seebeck_extractions.append({
                        'formula': nearest_formula,
                        'seebeck': seebeck_value,
                        'position': start_pos,
                        'context': text[max(0, start_pos-50):min(len(text), start_pos+100)]
                    })
    
    return seebeck_extractions

# -----------------------------
# GNN Regressor for Seebeck prediction
# -----------------------------
class GNNRegressor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, output_dim=1):
        super(GNNRegressor, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        self.fc = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = self.fc(x)
        return x.squeeze(-1)  # Remove last dimension for single output

def featurize_formulas(formulas, seebeck_values=None):
    """Convert formulas to graph data for GNN regression."""
    data_list = []
    valid_formulas = []
    valid_seebeck = [] if seebeck_values is not None else None

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
            continue

        try:
            comp = Composition(formula)
            el_amt_dict = comp.get_el_amt_dict()
            el_amt_dict = {k: max(1, round(v)) for k, v in el_amt_dict.items()}
            total_atoms = sum(el_amt_dict.values())
            if total_atoms < 2:
                continue

            species = []
            frac_coords = []
            pos = 0
            for el, amt in el_amt_dict.items():
                for _ in range(int(amt)):
                    species.append(el)
                    frac_coords.append([pos * 0.1, 0, 0])
                    pos += 1

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
            if not adjacency:
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
                continue

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_weights.unsqueeze(-1),
                y=torch.tensor([seebeck_values[i]], dtype=torch.float32) if seebeck_values is not None else None
            )

            data_list.append(data)
            valid_formulas.append(formula)
            if seebeck_values is not None:
                valid_seebeck.append(seebeck_values[i])

        except Exception:
            continue

    return data_list, valid_formulas, valid_seebeck

def train_gnn_regressor(formulas, seebeck_values):
    """Train GNN regressor for Seebeck coefficient prediction."""
    if not formulas or not seebeck_values:
        return None, None, {}

    data_list, valid_formulas, valid_seebeck = featurize_formulas(formulas, seebeck_values)
    if not data_list:
        return None, None, {}

    dataset = data_list
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GNNRegressor(input_dim=5, hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    model.train()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(200):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            update_log(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Best: {best_loss:.4f}")

        if patience_counter >= 20:
            update_log(f"Early stopping at epoch {epoch}")
            break

    # Save models
    model_files = {}
    save_formats = st.session_state.get('save_formats', ["pt", "h5"])
    
    if "pt" in save_formats:
        model_path = os.path.join(DB_DIR, "seebeck_gnn.pt")
        torch.save(model.state_dict(), model_path)
        model_files["seebeck_gnn.pt"] = model_path
    
    if "h5" in save_formats:
        h5_path = os.path.join(DB_DIR, "seebeck_gnn.h5")
        with h5py.File(h5_path, 'w') as f:
            model_group = f.create_group('gnn_regressor')
            for name, param in model.state_dict().items():
                model_group.create_dataset(name, data=param.cpu().numpy())
        model_files["seebeck_gnn.h5"] = h5_path

    update_log(f"Trained GNN regressor with {len(valid_formulas)} samples, RMSE: {np.sqrt(best_loss):.2f} μV/K")
    return model, None, model_files

def predict_seebeck_gnn(formula, model):
    """Predict Seebeck coefficient for a single formula using GNN."""
    if model is None:
        return None, "GNN model not trained"
    
    data_list, valid_formulas, _ = featurize_formulas([formula])
    if not data_list:
        return None, "Could not featurize formula"
    
    model.eval()
    data = data_list[0]
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    
    with torch.no_grad():
        prediction = model(data).item()
    
    return prediction, None

def standardize_material_formula(formula):
    """Standardize chemical formula."""
    if not formula or not validate_formula(formula):
        return None
    
    try:
        comp = Composition(formula)
        return comp.reduced_formula
    except:
        return None

# -----------------------------
# Main extraction function (renamed)
# -----------------------------
def extract_seebeck_coefficients(db_file, year_range=None):
    """Extract Seebeck coefficients from papers and link to formulas."""
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
    
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    nlp.add_pipe("formula_ner", last=True)
    
    all_extractions = []
    
    progress_bar = st.progress(0)
    for i, row in df.iterrows():
        update_progress(f"Processing paper {row['paper_id']} ({i+1}/{len(df)})")
        content = row[text_column]
        
        doc = nlp(content)
        formulas = [ent.text for ent in doc.ents if ent.label_ == "FORMULA"]
        
        seebeck_extractions = extract_seebeck_from_text(content, formulas)
        
        for extraction in seebeck_extractions:
            entry = {
                "paper_id": row["paper_id"],
                "title": row["title"],
                "formula": extraction["formula"],
                "seebeck": extraction["seebeck"],
                "context": extraction["context"]
            }
            if 'year' in row:
                entry['year'] = row['year']
            all_extractions.append(entry)
        
        progress_bar.progress((i + 1) / len(df))
    
    seebeck_df = pd.DataFrame(all_extractions)
    
    if seebeck_df.empty:
        return pd.DataFrame()
    
    # Cache results
    conn = sqlite3.connect(db_file)
    seebeck_df[["formula", "seebeck", "paper_id"] + (["year"] if 'year' in seebeck_df.columns else [])].to_sql(
        "seebeck_coefficients", conn, if_exists="replace", index=False
    )
    conn.close()
    
    # Train GNN
    formulas = seebeck_df["formula"].unique().tolist()
    seebeck_values = []
    for formula in formulas:
        formula_values = seebeck_df[seebeck_df["formula"] == formula]["seebeck"].mean()
        seebeck_values.append(formula_values)
    
    model, scaler, model_files = train_gnn_regressor(formulas, seebeck_values)
    st.session_state.gnn_model = model
    st.session_state.model_files = model_files
    st.session_state.seebeck_df = seebeck_df
    
    update_log(f"Extracted {len(seebeck_df)} Seebeck coefficients from {len(formulas)} unique materials")
    return seebeck_df

# -----------------------------
# Visualization functions
# -----------------------------
def plot_seebeck_data(df, top_n=20, year_range=None):
    if df.empty:
        return None, None, None
    
    if year_range and 'year' in df.columns:
        df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
    
    if df.empty:
        return None, None, None
    
    # Summary stats
    avg_seebeck = df.groupby("formula")["seebeck"].mean().reset_index()
    top_materials = avg_seebeck.nlargest(top_n, "seebeck")
    
    # Bar chart
    fig_bar = px.bar(
        top_materials, x="formula", y="seebeck",
        title=f"Top {top_n} Materials by Average Seebeck Coefficient (μV/K)",
        labels={"formula": "Formula", "seebeck": "Seebeck Coefficient (μV/K)"}
    )
    
    # Distribution
    fig_hist = px.histogram(
        df, x="seebeck", nbins=50,
        title="Distribution of Seebeck Coefficients",
        labels={"seebeck": "Seebeck Coefficient (μV/K)"}
    )
    
    # Scatter plot (formula complexity vs seebeck)
    df['atom_count'] = df['formula'].apply(lambda x: sum(Composition(x).get_el_amt_dict().values()))
    fig_scatter = px.scatter(
        df.groupby("formula").agg({"seebeck": "mean", "atom_count": "first"}).reset_index(),
        x="atom_count", y="seebeck",
        hover_data=["formula"],
        title="Seebeck vs Atomic Complexity",
        labels={"atom_count": "Number of Atoms", "seebeck": "Average Seebeck (μV/K)"}
    )
    
    return fig_bar, fig_hist, fig_scatter

# -----------------------------
# Streamlit UI (simplified)
# -----------------------------
def main():
    st.set_page_config(page_title="Seebeck Coefficient Extraction Tool", layout="wide")
    st.title("🔥 Seebeck Coefficient Extraction & Prediction Tool")
    st.markdown("Extract Seebeck coefficients from literature and predict for new materials using GNN regression")
    
    # Initialize session state
    for key in ["log_buffer", "error_summary", "progress_log", "save_formats", "gnn_model", "model_files", "seebeck_df"]:
        if key not in st.session_state:
            st.session_state[key] = [] if "log" in key or key == "error_summary" or key == "save_formats" else None
    
    st.session_state.save_formats = ["pt", "h5"]
    
    # Database selection (same as before)
    DB_DIR = os.path.dirname(os.path.abspath(__file__))
    db_files = glob.glob(os.path.join(DB_DIR, "*.db"))
    db_options = [os.path.basename(f) for f in db_files] + ["Upload a new .db file"]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        db_selection = st.selectbox("Select Database", db_options)
    with col2:
        if st.button("Extract Seebeck Coefficients"):
            pass  # Trigger extraction
    
    if db_selection != "Upload a new .db file":
        db_file = os.path.join(DB_DIR, db_selection)
        st.session_state.db_file = db_file
        
        if st.button("🚀 Extract Seebeck Coefficients", type="primary"):
            with st.spinner("Extracting Seebeck coefficients..."):
                year_range = st.session_state.get("year_range", (2000, 2025))
                df = extract_seebeck_coefficients(db_file, year_range)
                
                if not df.empty:
                    st.success(f"✅ Extracted {len(df)} Seebeck coefficients!")
                    st.session_state.seebeck_df = df
                    
                    # Visualizations
                    fig_bar, fig_hist, fig_scatter = plot_seebeck_data(df)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_bar, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig_hist, use_container_width=True)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Data table
                    st.dataframe(df[["formula", "seebeck", "paper_id", "context"]].head(100))
                    
                    # Download
                    csv = df.to_csv(index=False)
                    st.download_button("Download CSV", csv, "seebeck_coefficients.csv")
                else:
                    st.warning("No Seebeck coefficients found")
    
    # Formula prediction
    st.header("🔮 Predict Seebeck for New Formula")
    formula_input = st.text_input("Enter chemical formula (e.g., Bi2Te3, PbTe):")
    
    if st.button("Predict Seebeck Coefficient") and formula_input:
        with st.spinner("Predicting..."):
            if st.session_state.gnn_model is not None:
                prediction, error = predict_seebeck_gnn(formula_input, st.session_state.gnn_model)
                if error is None:
                    st.success(f"**Predicted Seebeck: {prediction:.1f} μV/K**")
                    
                    # Lookup existing data
                    if st.session_state.seebeck_df is not None:
                        matches = st.session_state.seebeck_df[
                            st.session_state.seebeck_df["formula"].str.lower() == formula_input.lower()
                        ]
                        if not matches.empty:
                            st.info(f"📚 Literature values: {matches['seebeck'].mean():.1f} ± {matches['seebeck'].std():.1f} μV/K")
                else:
                    st.error(error)
            else:
                st.error("Please extract Seebeck coefficients first to train the GNN model")

if __name__ == "__main__":
    main()
