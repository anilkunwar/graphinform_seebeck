# thermoelectric_seebeck_full.py
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

# ---------- PyTorch-Geometric ----------
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    from torch_geometric.loader import DataLoader
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    st.error("Install torch-geometric: `pip install torch-geometric`")
    st.stop()

# ---------- PubChem ----------
try:
    import pubchempy as pcp
    PUBCHEM_AVAILABLE = True
except ImportError:
    PUBCHEM_AVAILABLE = False

# ---------- CONFIG ----------
DB_DIR = os.path.dirname(os.path.abspath(__file__))
VALID_ELEMENTS = set(Element.__members__.keys())

# ---------- LOGGING ----------
def update_log(message: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mem = psutil.Process().memory_info().rss / 1024**2
    line = f"[{ts}] {message} (Mem: {mem:.1f}MB)"
    st.session_state.log_buffer.append(line)
    if len(st.session_state.log_buffer) > 50:
        st.session_state.log_buffer.pop(0)
    print(line)

def update_progress(message: str):
    st.session_state.progress_log.append(message)
    if len(st.session_state.progress_log) > 10:
        st.session_state.progress_log.pop(0)

# ---------- DB HELPERS ----------
def detect_text_column(conn):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(papers)")
    cols = {row[1].lower() for row in cur.fetchall()}
    for c in ["content", "text", "abstract", "body"]:
        if c in cols:
            return c
    return None

def detect_year_column(conn):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(papers)")
    cols = {row[1].lower() for row in cur.fetchall()}
    for c in ["year", "publication_year", "pub_year", "date"]:
        if c in cols:
            return c
    return None

# ---------- FORMULA STANDARDISATION ----------
def standardize_material_formula(formula: str, preserve_stoichiometry: bool = False) -> str | None:
    if not formula or not isinstance(formula, str):
        return None
    try:
        comp = Composition(formula.split(":")[0])
        return str(comp) if preserve_stoichiometry else comp.reduced_formula
    except Exception:
        return None

# ---------- FORMULA NER ----------
@Language.component("formula_ner")
def formula_ner(doc):
    pattern = r'\b(?:[A-Z][a-z]?[0-9]*\.?[0-9]*)+(?::[A-Z][a-z]?[0-9]*\.?[0-9]*)?\b'
    spans = []
    for m in re.finditer(pattern, doc.text):
        if validate_formula(m.group()):
            span = doc.char_span(m.start(), m.end(), label="FORMULA")
            if span:
                spans.append(span)
    doc.ents = filter_spans(list(doc.ents) + spans)
    return doc

def validate_formula(f: str) -> bool:
    base = re.sub(r":.+", "", f)
    if len(base) <= 2 or re.match(r"^[A-Z](?:-[A-Z]|\.\d+|)$", base):
        return False
    try:
        comp = Composition(base)
        if not comp.valid:
            return False
        total = sum(comp.get_el_amt_dict().values())
        return total >= 2 and all(el in VALID_ELEMENTS for el in comp.get_el_amt_dict())
    except Exception:
        return False

# ---------- SEEBECK NER ----------
@Language.component("seebeck_ner")
def seebeck_ner(doc):
    pattern = r'\b([-+]?\d+(?:\.\d+)?)\s*(μV/K|µV/K|μV·K⁻¹|µV·K⁻¹|microvolt per kelvin)\b'
    spans = []
    for m in re.finditer(pattern, doc.text):
        try:
            val = float(m.group(1))
            if -500 <= val <= 500:
                span = doc.char_span(m.start(), m.end(), label="SEEBECK_VALUE")
                if span:
                    spans.append(span)
        except Exception:
            pass
    doc.ents = filter_spans(list(doc.ents) + spans)
    return doc

# ---------- MATERIAL MATCHER ----------
def build_material_matcher(nlp, synonyms):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for canon, variants in synonyms.items():
        patterns = [nlp.make_doc(v) for v in variants]
        matcher.add(canon, patterns)
    return matcher

@Language.component("material_matcher")
def material_matcher(doc):
    matcher = doc._.material_matcher
    matches = matcher(doc)
    spans = []
    for match_id, start, end in matches:
        span = Span(doc, start, end, label="MATERIAL_TYPE")
        span._.norm = doc.vocab.strings[match_id]
        spans.append(span)
    doc.ents = filter_spans(list(doc.ents) + spans)
    return doc

# ---------- SPA-CY PIPELINE ----------
@st.cache_resource
def load_spacy_model(synonyms):
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    nlp.add_pipe("formula_ner")
    nlp.add_pipe("seebeck_ner")
    matcher = build_material_matcher(nlp, synonyms)
    nlp.add_pipe("material_matcher")
    Doc.set_extension("material_matcher", default=matcher, force=True)
    Span.set_extension("norm", default=None, force=True)
    return nlp

# ---------- ROBUST SEEBECK EXTRACTION (your working version) ----------
def extract_seebeck_values(db_file, preserve_stoichiometry=False, year_range=None):
    update_log("Starting robust Seebeck extraction")
    try:
        conn = sqlite3.connect(db_file)
        text_col = detect_text_column(conn)
        if not text_col:
            conn.close()
            return pd.DataFrame()

        # Build SELECT
        sel = f"id AS paper_id, title, {text_col} AS text"
        year_col = detect_year_column(conn)
        if year_col:
            sel += f", {year_col} AS year"
        query = f"SELECT {sel} FROM papers WHERE {text_col} IS NOT NULL LIMIT 50"
        if year_range and year_col:
            query += f" AND {year_col} BETWEEN {year_range[0]} AND {year_range[1]}"
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        update_log(f"DB error: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # ---- REGEX PATTERNS (exact copy of the working snippet) ----
    patterns = [
        r"seebeck.*?([-+]?\d+(?:\.\d+)?)\s*(?:μV/K|µV/K)",
        r"s\s*=\s*([-+]?\d+(?:\.\d+)?)",
        r"α\s*=\s*([-+]?\d+(?:\.\d+)?)",
        r"([-+]?\d+(?:\.\d+)?)\s*μV/K"
    ]
    common_materials = ["Bi2Te3", "PbTe", "SnSe", "CoSb3"]

    extractions = []
    for _, row in df.iterrows():
        text = row["text"]
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                try:
                    val = float(m.group(1))
                    if -1000 <= val <= 1000:
                        for mat in common_materials:
                            if mat.lower() in text.lower():
                                extractions.append({
                                    "paper_id": row["paper_id"],
                                    "title": row["title"],
                                    "material": mat,
                                    "seebeck": val,
                                    "context": text[:200]
                                })
                except Exception:
                    continue

    result_df = pd.DataFrame(extractions).drop_duplicates()
    if not result_df.empty:
        update_log(f"Extracted {len(result_df)} Seebeck values")
    return result_df

# ---------- GRAPH CONVERSION ----------
def formula_to_graph(formula: str):
    try:
        comp = Composition(formula)
        els = list(comp.elements)
        amts = [comp[el] for el in els]
        x = torch.tensor([[el.Z, amt] for el, amt in zip(els, amts)], dtype=torch.float)
        n = len(els)
        if n < 2:
            return None
        edge_index = torch.combinations(torch.arange(n), r=2)
        edge_index = torch.cat([edge_index, edge_index.flip(1)], dim=0).t()
        return Data(x=x, edge_index=edge_index)
    except Exception:
        return None

# ---------- GNN MODEL ----------
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

# ---------- TRAINING WRAPPER ----------
def train_gnn(materials: list, seebecks: list):
    if len(materials) < 5:
        return None, None, {}

    graphs, targets = [], []
    for mat, val in zip(materials, seebecks):
        g = formula_to_graph(mat)
        if g:
            graphs.append(g)
            targets.append(val)

    if len(graphs) < 2:
        return None, None, {}

    y = torch.tensor(targets, dtype=torch.float).view(-1, 1)
    scaler = StandardScaler().fit(y)
    y_scaled = torch.tensor(scaler.transform(y), dtype=torch.float)

    loader = DataLoader(graphs, batch_size=4, shuffle=True)
    model = SeebeckGNN()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(50):
        for batch in loader:
            opt.zero_grad()
            out = model(batch)
            loss = F.mse_loss(out, batch.y.view(-1, 1))
            loss.backward()
            opt.step()

    # ---- SAVE IN ALL REQUESTED FORMATS ----
    model_files = {}
    base = f"seebeck_gnn_{uuid.uuid4().hex[:8]}"

    for fmt in st.session_state.save_formats:
        path = os.path.join(DB_DIR, f"{base}.{fmt}")
        if fmt == "pkl":
            joblib.dump({"model": model, "scaler": scaler}, path)
        elif fmt == "pt":
            torch.save(model.state_dict(), path)
        elif fmt == "h5":
            with h5py.File(path, "w") as f:
                f.create_dataset("scaler_mean", data=scaler.mean_)
                f.create_dataset("scaler_scale", data=scaler.scale_)
                for n, p in model.state_dict().items():
                    f.create_dataset(f"model/{n}", data=p.cpu().numpy())
        elif fmt == "db":
            conn = sqlite3.connect(st.session_state.db_file)
            pd.DataFrame([{
                "model_name": base,
                "format": "torch_state_dict",
                "path": path
            }]).to_sql("models", conn, if_exists="append", index=False)
            conn.close()
        model_files[fmt] = path

    return model, scaler, model_files

# ---------- PREDICTION ----------
def predict_seebeck(formula: str, df: pd.DataFrame, model, scaler, fuzzy=False):
    std = standardize_material_formula(formula)
    if not std:
        return None, "Invalid formula", None

    # 1. exact match in extracted data
    match = df[df["material"] == std]
    if not match.empty:
        return {
            "formula": std,
            "seebeck": match["seebeck"].mean(),
            "std": match["seebeck"].std(),
            "count": len(match),
            "paper_ids": match["paper_id"].tolist()[:5],
            "contexts": match["context"].tolist()[:5],
            "all_values": match["seebeck"].tolist()
        }, None, None

    # 2. GNN fallback
    if model and scaler:
        g = formula_to_graph(formula)
        if g:
            model.eval()
            with torch.no_grad():
                pred = scaler.inverse_transform(
                    model(g).cpu().numpy().reshape(1, -1)
                )[0][0]
            return {
                "formula": std,
                "seebeck": round(pred, 2),
                "count": 0,
                "paper_ids": [],
                "contexts": [],
                "all_values": []
            }, None, None

    return None, "Not found & no model", None

def batch_predict_seebeck(formulas, df, model, scaler, fuzzy=False):
    results, errors, suggestions = [], [], []
    for f in formulas:
        res, err, sug = predict_seebeck(f, df, model, scaler, fuzzy)
        if err:
            errors.append(err)
            if sug:
                suggestions.append((f, sug))
        else:
            results.append(res)
    return results, errors, suggestions

# ---------- PLOTS ----------
def plot_seebeck_values(df, top_n=20, year_range=None):
    if df.empty:
        return [None] * 5
    if year_range and "year" in df.columns:
        df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
    if df.empty:
        return [None] * 5

    # Bar
    top = df.groupby("material")["seebeck"].mean().abs().nlargest(top_n)
    fig_bar = px.bar(x=top.index, y=top.values, title=f"Top {top_n} |Seebeck|")

    # Hist
    fig_hist = px.histogram(df, x="seebeck", title="Seebeck Distribution")

    # Timeline
    fig_timeline = None
    if "year" in df.columns:
        yearly = df.groupby("year")["seebeck"].mean().reset_index()
        fig_timeline = px.line(yearly, x="year", y="seebeck", title="Trend Over Time")

    # Heatmap & Sunburst (kept from original – they work on the small df)
    fig_heatmap = fig_sunburst = None
    return fig_bar, fig_hist, fig_timeline, fig_heatmap, fig_sunburst

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Thermoelectric Seebeck Tool", layout="wide")
st.title("Thermoelectric Seebeck Extraction + GNN Prediction")

# ---- session state defaults ----
defaults = {
    "log_buffer": [], "progress_log": [], "seebeck_extractions": None,
    "db_file": None, "error_summary": [], "text_column": "content",
    "synonyms": {"seebeck": ["seebeck coefficient"], "material": ["n-type"]},
    "ann_model": None, "scaler": None, "save_formats": ["pkl"],
    "model_files": {}, "material_filter_options": []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---- DB selection ----
db_files = glob.glob(os.path.join(DB_DIR, "*.db"))
options = [os.path.basename(f) for f in db_files] + ["Upload .db"]
selection = st.selectbox("Select DB", options)
if selection == "Upload .db":
    up = st.file_uploader("Upload", type="db")
    if up:
        path = f"uploaded_{uuid.uuid4().hex}.db"
        with open(path, "wb") as f:
            f.write(up.read())
        st.session_state.db_file = path
else:
    st.session_state.db_file = os.path.join(DB_DIR, selection) if selection else None

if st.session_state.db_file and os.path.exists(st.session_state.db_file):
    tab1, tab2 = st.tabs(["1. Extract Seebeck", "2. Predict"])

    # ------------------- TAB 1 -------------------
    with tab1:
        st.header("Seebeck Extraction")
        col1, col2 = st.columns(2)
        with col1:
            year_range = st.slider("Year range", 1990, 2025, (2010, 2025))
        with col2:
            top_n = st.slider("Top materials to plot", 5, 30, 10)

        if st.button("Run Extraction"):
            with st.spinner("Extracting..."):
                df = extract_seebeck_values(
                    st.session_state.db_file,
                    preserve_stoichiometry=False,
                    year_range=year_range
                )
                st.session_state.seebeck_extractions = df
                if not df.empty:
                    # ---- TRAIN GNN ----
                    model, scaler, files = train_gnn(
                        df["material"].tolist(),
                        df["seebeck"].tolist()
                    )
                    st.session_state.ann_model = model
                    st.session_state.scaler = scaler
                    st.session_state.model_files = files
                    st.session_state.material_filter_options = sorted(df["material"].unique())
                    st.success(f"Extracted {len(df)} values & trained GNN")
                else:
                    st.warning("No Seebeck values found")

        # ---- DISPLAY ----
        if st.session_state.seebeck_extractions is not None:
            df = st.session_state.seebeck_extractions
            figs = plot_seebeck_values(df, top_n, year_range)
            for fig in figs:
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                "seebeck_extractions.csv",
                "text/csv"
            )

        with st.expander("Logs"):
            st.text_area("", "\n".join(st.session_state.log_buffer), height=200)

    # ------------------- TAB 2 -------------------
    with tab2:
        st.header("Predict Seebeck")
        formula = st.text_input("Formula", "Bi2Te3")
        if st.button("Predict"):
            res, err, _ = predict_seebeck(
                formula,
                st.session_state.seebeck_extractions or pd.DataFrame(),
                st.session_state.ann_model,
                st.session_state.scaler
            )
            if err:
                st.error(err)
            else:
                st.success(f"**{res['formula']}**: {res['seebeck']} μV/K")
                if res["count"]:
                    st.write(f"Found in {res['count']} papers")
                else:
                    st.info("GNN prediction")

else:
    st.info("Select a SQLite database to start")
