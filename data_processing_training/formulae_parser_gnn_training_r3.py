# thermoelectric_seebeck_robust_fixed.py
import os
import sqlite3
import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
import uuid
import psutil
from datetime import datetime
import numpy as np
import glob
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element

# PyTorch Geometric
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    from torch_geometric.loader import DataLoader
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    st.error("Install: `pip install torch-geometric`")
    st.stop()

# ========================
# CONFIG & LOGGING
# ========================
DB_DIR = os.path.dirname(os.path.abspath(__file__))
VALID_ELEMENTS = set(Element.__members__.keys())
COMMON_MATERIALS = ["Bi2Te3", "PbTe", "SnSe", "CoSb3", "SiGe", "Mg2Si", "Zn4Sb3", "Cu2Se"]

def update_log(message):
    ts = datetime.now().strftime("%H:%M:%S")
    mem = psutil.Process().memory_info().rss / 1024**2
    msg = f"[{ts}] {message} (Mem: {mem:.1f}MB)"
    st.session_state.log_buffer.append(msg)
    if len(st.session_state.log_buffer) > 50: st.session_state.log_buffer.pop(0)
    print(msg)

def update_progress(message):
    st.session_state.progress_log.append(message)
    if len(st.session_state.progress_log) > 10: st.session_state.progress_log.pop(0)

# ========================
# COLUMN DETECTION
# ========================
def detect_text_column(conn):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(papers)")
    cols = [row[1].lower() for row in cur.fetchall()]
    for c in ['content', 'text', 'abstract', 'body']: 
        if c in cols: return c
    return None

def detect_year_column(conn):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(papers)")
    cols = [row[1].lower() for row in cur.fetchall()]
    for c in ['year', 'publication_year', 'pub_year', 'date']: 
        if c in cols: return c
    return None

# ========================
# STANDARDIZE FORMULA
# ========================
def standardize_material_formula(formula, preserve=False):
    try:
        comp = Composition(formula)
        return str(comp) if preserve else comp.reduced_formula
    except:
        return formula

# ========================
# ROBUST SEEBECK EXTRACTION (YOUR WORKING CODE)
# ========================
def extract_seebeck_values(db_file, preserve_stoichiometry=False, year_range=None):
    update_log("Starting robust Seebeck extraction...")
    try:
        conn = sqlite3.connect(db_file)
        text_col = detect_text_column(conn)
        if not text_col:
            update_log("No text column found")
            conn.close()
            return pd.DataFrame()

        year_col = detect_year_column(conn)
        query = f"SELECT id AS paper_id, title, {text_col} AS text"
        if year_col: query += f", {year_col} AS year"
        query += f" FROM papers WHERE {text_col} IS NOT NULL AND {text_col} NOT LIKE 'Error%' LIMIT 100"
        if year_range and year_col:
            query += f" AND {year_col} BETWEEN {year_range[0]} AND {year_range[1]}"
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        update_log(f"DB Error: {e}")
        return pd.DataFrame()

    if df.empty:
        update_log("No papers")
        return pd.DataFrame()

    patterns = [
        r"seebeck.*?([-+]?\d+(?:\.\d+)?)\s*(?:μV/K|µV/K)",
        r"s\s*=\s*([-+]?\d+(?:\.\d+)?)",
        r"α\s*=\s*([-+]?\d+(?:\.\d+)?)",
        r"([-+]?\d+(?:\.\d+)?)\s*μV/K"
    ]

    extractions = []
    progress = st.progress(0)

    for i, row in df.iterrows():
        progress.progress((i+1)/len(df))
        text = str(row['text'])[:500000].lower()
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                try:
                    val = float(match.group(1))
                    if -1000 <= val <= 1000:
                        for mat in COMMON_MATERIALS:
                            if mat.lower() in text:
                                entry = {
                                    "paper_id": row["paper_id"],
                                    "title": row["title"],
                                    "material": standardize_material_formula(mat, preserve_stoichiometry),
                                    "seebeck": round(val, 2),
                                    "context": text[max(0, match.start()-50):match.end()+50]
                                }
                                if 'year' in row: entry["year"] = row["year"]
                                extractions.append(entry)
                except: continue

    result_df = pd.DataFrame(extractions).drop_duplicates()
    if not result_df.empty:
        update_log(f"EXTRACTED {len(result_df)} VALUES!")
        # Cache
        try:
            conn = sqlite3.connect(db_file)
            result_df[["material", "seebeck"] + (["year"] if 'year' in result_df.columns else [])].to_sql(
                "standardized_formulas", conn, if_exists="replace", index=False)
            conn.close()
        except: pass
    else:
        update_log("No values found")
    return result_df

# ========================
# GNN: GRAPH FROM FORMULA
# ========================
def formula_to_graph(formula):
    try:
        comp = Composition(formula)
        els = list(comp.elements)
        amts = [comp[el] for el in els]
        x = torch.tensor([[el.Z, amt] for el, amt in zip(els, amts)], dtype=torch.float)
        n = len(els)
        if n < 2: return None
        edge_index = torch.combinations(torch.arange(n), r=2)
        edge_index = torch.cat([edge_index, edge_index.flip(1)], dim=0).t()
        return Data(x=x, edge_index=edge_index)
    except: return None

# ========================
# GNN MODEL
# ========================
class SeebeckGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(2, 64)
        self.conv2 = GCNConv(64, 64)
        self.lin = nn.Linear(64, 1)
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        return self.lin(x.mean(dim=0))

# ========================
# TRAIN GNN
# ========================
def train_gnn(df):
    if len(df) < 5:
        update_log("Not enough data for GNN")
        return None, None, {}
    graphs, targets = [], []
    for _, r in df.iterrows():
        g = formula_to_graph(r["material"])
        if g:
            graphs.append(g)
            targets.append(r["seebeck"])
    if len(graphs) < 2: return None, None, {}
    y = torch.tensor(targets, dtype=torch.float).view(-1,1)
    scaler = StandardScaler().fit(y)
    y_scaled = torch.tensor(scaler.transform(y), dtype=torch.float)
    loader = DataLoader(graphs, batch_size=4, shuffle=True)
    model = SeebeckGNN()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    update_log("Training GNN...")
    for epoch in range(50):
        for batch in loader:
            opt.zero_grad()
            out = model(batch)
            loss = F.mse_loss(out, batch.y.view(-1,1))
            loss.backward()
            opt.step()
    update_log("GNN trained!")
    return model, scaler, {}

# ========================
# PREDICT
# ========================
def predict_seebeck(formula, df, model=None, scaler=None):
    std = standardize_material_formula(formula)
    match = df[df['material'] == std]
    if not match.empty:
        return {
            "formula": std,
            "seebeck": match['seebeck'].mean(),
            "std": match['seebeck'].std(),
            "count": len(match),
            "paper_ids": match['paper_id'].tolist()[:5],
            "contexts": match['context'].tolist()[:5],
            "all_values": match['seebeck'].tolist()
        }
    elif model and scaler:
        g = formula_to_graph(formula)
        if g:
            model.eval()
            with torch.no_grad():
                pred = scaler.inverse_transform(model(g).cpu().numpy().reshape(1,-1))[0][0]
            return {"formula": std, "seebeck": round(pred, 2), "count": 0, "paper_ids": [], "contexts": [], "all_values": []}
    return None

# ========================
# PLOT (KEEP ALL)
# ========================
def plot_seebeck_values(df, top_n=10, year_range=None):
    if df.empty: return [None]*5
    if year_range and 'year' in df.columns:
        df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    if df.empty: return [None]*5

    top = df.groupby('material')['seebeck'].mean().abs().nlargest(top_n)
    fig_bar = px.bar(x=top.index, y=top.values, title=f"Top {top_n} Materials")
    fig_hist = px.histogram(df, x='seebeck', title="Distribution")

    fig_timeline = None
    if 'year' in df.columns:
        yearly = df.groupby('year')['seebeck'].mean().reset_index()
        fig_timeline = px.line(yearly, x='year', y='seebeck', title="Trend Over Time")

    fig_heatmap = fig_sunburst = None
    return fig_bar, fig_hist, fig_timeline, fig_heatmap, fig_sunburst

# ========================
# STREAMLIT UI (FULL FEATURES)
# ========================
st.set_page_config(page_title="Seebeck GNN", layout="wide")
st.title("Seebeck Extractor + GNN Predictor")

# Session state
for k, v in [
    ("log_buffer", []), ("progress_log", []), ("seebeck_extractions", None),
    ("db_file", None), ("ann_model", None), ("scaler", None), ("model_files", {}),
    ("synonyms", {"seebeck": ["seebeck coefficient"], "material": ["n-type"]}),
    ("material_filter_options", []), ("error_summary", [])
]:
    if k not in st.session_state: st.session_state[k] = v

# DB Selection
db_files = glob.glob(os.path.join(DB_DIR, "*.db"))
options = ["Upload .db"] + [os.path.basename(f) for f in db_files]
choice = st.selectbox("Select Database", options)

if choice == "Upload .db":
    up = st.file_uploader("Upload DB", type="db")
    if up:
        path = f"uploaded_{uuid.uuid4().hex}.db"
        with open(path, "wb") as f: f.write(up.read())
        st.session_state.db_file = path
else:
    st.session_state.db_file = os.path.join(DB_DIR, choice)

if st.session_state.db_file and os.path.exists(st.session_state.db_file):
    tab1, tab2 = st.tabs(["1. Extract & Train", "2. Predict"])

    with tab1:
        st.header("Extract Seebeck & Train GNN")
        col1, col2 = st.columns(2)
        with col1: year_range = st.slider("Year Range", 1990, 2025, (2010, 2025))
        with col2: preserve = st.checkbox("Preserve Stoichiometry")

        if st.button("Extract + Train GNN"):
            with st.spinner("Extracting..."):
                df = extract_seebeck_values(st.session_state.db_file, preserve, year_range)
                st.session_state.seebeck_extractions = df
                if not df.empty:
                    st.success(f"Extracted {len(df)} values!")
                    st.session_state.material_filter_options = sorted(df["material"].unique())
                    model, scaler, files = train_gnn(df)
                    st.session_state.ann_model = model
                    st.session_state.scaler = scaler
                    st.session_state.model_files = files
                else:
                    st.warning("No data extracted")

        if st.session_state.seebeck_extractions is not None and not st.session_state.seebeck_extractions.empty:
            df = st.session_state.seebeck_extractions
            material_filter = st.multiselect("Filter Materials", st.session_state.material_filter_options)
            filtered = df if not material_filter else df[df["material"].isin(material_filter)]

            st.subheader("Results")
            col1, col2 = st.columns(2)
            with col1: st.metric("Total", len(filtered))
            with col2: st.metric("Avg Seebeck", f"{filtered['seebeck'].mean():.2f} μV/K")

            fig_bar, fig_hist, fig_timeline, _, _ = plot_seebeck_values(filtered)
            if fig_bar: st.plotly_chart(fig_bar, use_container_width=True)
            if fig_hist: st.plotly_chart(fig_hist, use_container_width=True)
            if fig_timeline: st.plotly_chart(fig_timeline, use_container_width=True)

            st.download_button("Download CSV", filtered.to_csv(index=False), "seebeck.csv")

    with tab2:
        st.header("Predict Seebeck")
        formula = st.text_input("Enter Formula", "Bi2Te3")
        if st.button("Predict"):
            res = predict_seebeck(formula, st.session_state.seebeck_extractions or pd.DataFrame(),
                                  st.session_state.ann_model, st.session_state.scaler)
            if res:
                st.success(f"**{res['formula']}**: {res['seebeck']} μV/K")
                if res['count'] > 0:
                    st.write(f"Found in {res['count']} papers")
                else:
                    st.write("GNN Prediction")
            else:
                st.error("Invalid formula")

    with st.expander("Logs"):
        st.text("\n".join(st.session_state.log_buffer))
else:
    st.info("Select a database")
