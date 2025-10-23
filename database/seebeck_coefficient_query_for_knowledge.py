import arxiv
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
import os
import re
import sqlite3
from datetime import datetime
import logging
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax
from collections import Counter
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed
import zipfile
import gc
import psutil
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import tempfile
import json

# ===== CLOUD-OPTIMIZED CONFIGURATION =====
if os.path.exists("/tmp"):  # Cloud environments typically have /tmp
    DB_DIR = "/tmp"
else:
    DB_DIR = os.path.join(os.path.expanduser("~"), "Desktop")

# Ensure directories exist
os.makedirs(DB_DIR, exist_ok=True)
pdf_dir = os.path.join(DB_DIR, "pdfs")
os.makedirs(pdf_dir, exist_ok=True)

METADATA_DB_FILE = os.path.join(DB_DIR, "seebeck_metadata.db")
UNIVERSE_DB_FILE = os.path.join(DB_DIR, "seebeck_universe.db")

# Initialize logging
logging.basicConfig(filename=os.path.join(DB_DIR, 'seebeck_query.log'), 
                   level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Streamlit app
st.set_page_config(page_title="Seebeck Coefficient Query Tool", layout="wide")
st.title("Seebeck Coefficient Query Tool with SciBERT")
st.markdown("""
This tool queries arXiv for papers on **Seebeck coefficient**, focusing on aspects such as **thermopower**, **power factor**, **ZT**, **figure of merit**, **thermoelectric materials**, **band gap**, **electrical conductivity**, **thermal conductivity**, **carrier concentration**, **carrier mobility**, **p-type**, **n-type**. It uses SciBERT to prioritize relevant abstracts (>30% relevance) and stores metadata in `seebeck_metadata.db` and full PDF text in `seebeck_universe.db` for fallback searches. PDFs and database files are stored individually and can be downloaded as a ZIP file.
""")

# Dependency check
st.sidebar.header("Setup")
st.sidebar.markdown("""
**Dependencies**:
- `arxiv`, `pymupdf`, `pandas`, `streamlit`, `transformers`, `torch`, `scipy`, `numpy`, `tenacity`, `requests`, `psutil`
- Install: `pip install arxiv pymupdf pandas streamlit transformers torch scipy numpy tenacity requests psutil`
""")

# ===== RESOURCE MANAGEMENT =====
def check_memory_usage():
    """Check current memory usage"""
    try:
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        return memory_usage
    except:
        return 0

def cleanup_memory():
    """Clean up memory and GPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def system_health_check():
    """Check system resources before processing"""
    try:
        memory_usage = check_memory_usage()
        disk_usage = psutil.disk_usage(DB_DIR)
        disk_free_gb = disk_usage.free / (1024**3)
        
        update_log(f"System health - Memory: {memory_usage:.1f}MB, Disk free: {disk_free_gb:.1f}GB")
        
        if memory_usage > 1500:  # 1.5GB
            st.warning(f"High memory usage ({memory_usage:.1f}MB), processing may be slow")
            cleanup_memory()
        if disk_free_gb < 0.5:  # 500MB free space
            st.error(f"Low disk space ({disk_free_gb:.1f}GB), some operations may fail")
            return False
        return True
    except Exception as e:
        update_log(f"Health check warning: {str(e)}")
        return True  # Continue anyway

def create_retry_session():
    """Create HTTP session with retry strategy"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def limit_pdf_processing(papers, max_pdfs=10):
    """Limit the number of PDFs processed in cloud environment"""
    if len(papers) > max_pdfs and os.path.exists("/tmp"):  # Cloud environment
        st.warning(f"Cloud environment: Limiting to {max_pdfs} PDF downloads")
        return papers[:max_pdfs]
    return papers

# ===== SESSION STATE MANAGEMENT =====
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "current_progress" not in st.session_state:
    st.session_state.current_progress = 0
if "download_files" not in st.session_state:
    st.session_state.download_files = {"pdf_paths": [], "zip_path": None}
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "relevant_papers" not in st.session_state:
    st.session_state.relevant_papers = None
if "search_params" not in st.session_state:
    st.session_state.search_params = None

def reset_processing():
    """Reset processing state"""
    st.session_state.processing = False
    st.session_state.current_progress = 0

def reset_downloads():
    """Reset download-related state and clear cache"""
    st.session_state.download_files = {"pdf_paths": [], "zip_path": None}
    st.session_state.search_results = None
    st.session_state.relevant_papers = None
    query_arxiv.clear()  # Clear the arXiv query cache
    cleanup_memory()
    update_log("Download state and cache cleared")

def update_log(message):
    """Update log with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.log_buffer.append(log_entry)
    if len(st.session_state.log_buffer) > 30:
        st.session_state.log_buffer.pop(0)
    logging.info(message)

# ===== MODEL LOADING =====
@st.cache_resource
def load_scibert_model():
    """Load SciBERT model with caching"""
    try:
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        model = AutoModelForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased")
        model.eval()
        update_log("SciBERT model loaded successfully")
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load SciBERT: {e}. Install: `pip install transformers torch`")
        st.stop()

scibert_tokenizer, scibert_model = load_scibert_model()

# ===== DATABASE OPERATIONS =====
def initialize_db(db_file):
    """Initialize database schema"""
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        if 'universe' in db_file:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    authors TEXT,
                    year INTEGER,
                    content TEXT
                )
            """)
        else:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    authors TEXT,
                    year INTEGER,
                    categories TEXT,
                    abstract TEXT,
                    pdf_url TEXT,
                    download_status TEXT,
                    matched_terms TEXT,
                    relevance_prob REAL,
                    pdf_path TEXT,
                    content TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parameters (
                    paper_id TEXT,
                    entity_text TEXT,
                    entity_label TEXT,
                    value REAL,
                    unit TEXT,
                    context TEXT,
                    phase TEXT,
                    score REAL,
                    co_occurrence BOOLEAN,
                    FOREIGN KEY (paper_id) REFERENCES papers(id)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_paper_id ON parameters(paper_id)")
        conn.commit()
        conn.close()
        update_log(f"Initialized database schema for {db_file}")
    except Exception as e:
        update_log(f"Failed to initialize {db_file}: {str(e)}")
        st.error(f"Failed to initialize {db_file}: {str(e)}")

initialize_db(METADATA_DB_FILE)
initialize_db(UNIVERSE_DB_FILE)

# ===== KEY TERMS AND SCORING =====
KEY_TERMS = [
    "seebeck coefficient", "thermopower", "seebeck", "power factor", "zt", "figure of merit",
    "thermoelectric", "thermoelectric material", "band gap", "electrical conductivity",
    "thermal conductivity", "carrier concentration", "carrier mobility", "p-type", "n-type"
]

def score_abstract_with_scibert(abstract):
    """Score abstract relevance using SciBERT"""
    try:
        inputs = scibert_tokenizer(abstract, return_tensors="pt", truncation=True, max_length=512, padding=True, return_attention_mask=True)
        with torch.no_grad():
            outputs = scibert_model(**inputs, output_attentions=True)
        logits = outputs.logits.numpy()
        probs = softmax(logits, axis=1)
        relevance_prob = probs[0][1]
        tokens = scibert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        keyword_indices = [i for i, token in enumerate(tokens) if any(kw.lower() in token.lower() for kw in KEY_TERMS)]
        if keyword_indices:
            attentions = outputs.attentions[-1][0, 0].numpy()
            attn_score = np.sum(attentions[keyword_indices, :]) / len(keyword_indices)
            if attn_score > 0.1 and relevance_prob < 0.5:
                relevance_prob = min(relevance_prob + 0.2 * len(keyword_indices), 1.0)
        update_log(f"SciBERT scored abstract: {relevance_prob:.3f} (keywords matched: {len(keyword_indices)})")
        return relevance_prob
    except Exception as e:
        update_log(f"SciBERT scoring failed: {str(e)}")
        # Fallback scoring
        abstract_lower = abstract.lower()
        word_counts = Counter(re.findall(r'\b\w+\b', abstract_lower))
        total_words = sum(word_counts.values())
        score = sum(word_counts.get(kw.lower(), 0) for kw in KEY_TERMS) / (total_words + 1e-6)
        max_possible_score = len(KEY_TERMS) / 10
        relevance_prob = min(score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0
        update_log(f"Fallback scoring: {relevance_prob:.3f}")
        return relevance_prob

# ===== PDF PROCESSING =====
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        update_log(f"PDF extraction failed for {pdf_path}: {str(e)}")
        return f"Error: {str(e)}"

def update_db_content(db_file, paper_id, content):
    """Update content in database"""
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("UPDATE papers SET content = ? WHERE id = ?", (content, paper_id))
        if cursor.rowcount == 0:
            if 'universe' in db_file:
                cursor.execute("""
                    INSERT INTO papers (id, title, authors, year, content)
                    VALUES (?, ?, ?, ?, ?)
                """, (paper_id, "Unknown", "Unknown", 0, content))
                update_log(f"Inserted placeholder for {paper_id} in {db_file}")
            else:
                update_log(f"Skipped inserting {paper_id} in {db_file} as metadata not present")
        else:
            update_log(f"Updated content for {paper_id} in {db_file}")
        conn.commit()
        conn.close()
    except Exception as e:
        update_log(f"Failed to update content in {db_file} for {paper_id}: {str(e)}")

# ===== BATCH PROCESSING =====
def batch_convert_pdfs():
    """Batch convert existing PDFs to databases"""
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if not pdf_files:
        update_log("No PDFs found in directory.")
        return
    
    if not system_health_check():
        st.error("System health check failed. Cannot process PDFs.")
        return
        
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, filename in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_dir, filename)
        paper_id = filename[:-4]
        status_text.text(f"Processing {i+1}/{len(pdf_files)}: {filename}")
        
        text = extract_text_from_pdf(pdf_path)
        if not text.startswith("Error"):
            update_db_content(METADATA_DB_FILE, paper_id, text)
            update_db_content(UNIVERSE_DB_FILE, paper_id, text)
        else:
            update_log(text)
            
        progress_bar.progress((i + 1) / len(pdf_files))
        time.sleep(0.1)  # Small delay to avoid overwhelming
        
        # Clean memory every 5 files
        if i % 5 == 0:
            cleanup_memory()
    
    status_text.empty()
    cleanup_memory()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def create_universe_db(paper, db_file=UNIVERSE_DB_FILE):
    """Create universe database entry"""
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                content TEXT
            )
        """)
        cursor.execute("""
            INSERT OR REPLACE INTO papers (id, title, authors, year, content)
            VALUES (?, ?, ?, ?, ?)
        """, (
            paper["id"],
            paper.get("title", ""),
            paper.get("authors", "Unknown"),
            paper.get("year", 0),
            paper.get("content", "No text extracted")
        ))
        conn.commit()
        conn.close()
        update_log(f"Updated {db_file} with paper {paper['id']}")
        return db_file
    except Exception as e:
        update_log(f"Error updating {db_file}: {str(e)}")
        raise

# ===== DATA STORAGE =====
def save_to_sqlite(papers_df, params_list, metadata_db_file=METADATA_DB_FILE):
    """Save data to SQLite database"""
    try:
        initialize_db(metadata_db_file)
        conn = sqlite3.connect(metadata_db_file)
        papers_df.to_sql("papers", conn, if_exists="replace", index=False)
        params_df = pd.DataFrame(params_list)
        if not params_df.empty:
            params_df.to_sql("parameters", conn, if_exists="append", index=False)
        conn.close()
        update_log(f"Saved {len(papers_df)} papers and {len(params_list)} parameters to {metadata_db_file}")
        return f"Saved to {metadata_db_file}"
    except Exception as e:
        update_log(f"SQLite save failed: {str(e)}")
        return f"Failed to save to SQLite: {str(e)}"

# ===== ARXIV QUERY =====
@st.cache_data(hash_funcs={list: lambda x: str(x), tuple: lambda x: str(x)}, ttl=3600)
def query_arxiv(query, categories, max_results, start_year, end_year):
    """Query arXiv for papers"""
    try:
        query_terms = query.strip().split()
        formatted_terms = [term.strip('"').replace(" ", "+") for term in query_terms]
        api_query = " ".join(formatted_terms)
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=api_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        papers = []
        for result in client.results(search):
            if any(cat in result.categories for cat in categories) and start_year <= result.published.year <= end_year:
                abstract = result.summary.lower()
                title = result.title.lower()
                query_words = set(word.lower().strip('"') for word in query_terms)
                matched_terms = [word for word in query_words if word in abstract or word in title]
                if not matched_terms:
                    continue
                relevance_prob = score_abstract_with_scibert(result.summary)
                abstract_highlighted = abstract
                for term in matched_terms:
                    abstract_highlighted = re.sub(r'\b{}\b'.format(re.escape(term)), f'<b style="color: orange">{term}</b>', abstract_highlighted, flags=re.IGNORECASE)
                
                papers.append({
                    "id": result.entry_id.split('/')[-1],
                    "title": result.title,
                    "authors": ", ".join([author.name for author in result.authors]),
                    "year": result.published.year,
                    "categories": ", ".join(result.categories),
                    "abstract": result.summary,
                    "abstract_highlighted": abstract_highlighted,
                    "pdf_url": result.pdf_url,
                    "download_status": "Not downloaded",
                    "matched_terms": ", ".join(matched_terms) if matched_terms else "None",
                    "relevance_prob": round(relevance_prob * 100, 2),
                    "pdf_path": None,
                    "content": None
                })
            if len(papers) >= max_results:
                break
        papers = sorted(papers, key=lambda x: x["relevance_prob"], reverse=True)
        update_log(f"Found {len(papers)} papers")
        return papers
    except Exception as e:
        update_log(f"arXiv query failed: {str(e)}")
        st.error(f"Error querying arXiv: {str(e)}. Try simplifying the query.")
        return []

# ===== PDF DOWNLOAD =====
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def download_pdf_and_extract(pdf_url, paper_id, paper_metadata):
    """Download PDF and extract text"""
    pdf_path = os.path.join(pdf_dir, f"{paper_id}.pdf")
    session = create_retry_session()
    try:
        response = session.get(pdf_url, timeout=30)
        response.raise_forcelist()
        
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
            
        file_size = os.path.getsize(pdf_path) / 1024
        text = extract_text_from_pdf(pdf_path)
        
        if not text.startswith("Error"):
            paper_data = {
                "id": paper_id,
                "title": paper_metadata.get("title", ""),
                "authors": paper_metadata.get("authors", "Unknown"),
                "year": paper_metadata.get("year", 0),
                "content": text
            }
            create_universe_db(paper_data)
            update_db_content(METADATA_DB_FILE, paper_id, text)
            update_log(f"Downloaded and extracted text for paper {paper_id} ({file_size:.2f} KB)")
            return f"Downloaded ({file_size:.2f} KB)", pdf_path, text
        else:
            update_log(f"Text extraction failed for {paper_id}: {text}")
            return f"Failed: {text}", None, text
            
    except Exception as e:
        update_log(f"PDF download failed for {paper_id}: {str(e)}")
        return f"Failed: {str(e)}", None, f"Error: {str(e)}"
    finally:
        session.close()

# ===== FILE MANAGEMENT =====
def create_pdf_zip(pdf_paths):
    """Create ZIP file of PDFs"""
    zip_path = os.path.join(DB_DIR, "seebeck_pdfs.zip")
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for pdf_path in pdf_paths:
                if pdf_path and os.path.exists(pdf_path):
                    zipf.write(pdf_path, os.path.basename(pdf_path))
        update_log(f"Created ZIP file at {zip_path}")
        return zip_path
    except Exception as e:
        update_log(f"Failed to create ZIP file: {str(e)}")
        return None

def read_file_for_download(file_path):
    """Read file for download and ensure handle is closed"""
    try:
        with open(file_path, "rb") as f:
            return f.read()
    except Exception as e:
        update_log(f"Failed to read file {file_path} for download: {str(e)}")
        return None

# ===== STREAMLIT UI =====
st.header("arXiv Query for Seebeck Coefficient")
st.markdown("Search for abstracts on **Seebeck coefficient**, including **thermopower**, **power factor**, **ZT**, **figure of merit**, **thermoelectric materials**, **band gap**, **electrical conductivity**, **thermal conductivity**, **carrier concentration**, **carrier mobility**, **p-type**, **n-type** using SciBERT.")

log_container = st.empty()
def display_logs():
    """Display processing logs"""
    log_container.text_area("Processing Logs", "\n".join(st.session_state.log_buffer), height=200)

# Sidebar configuration
with st.sidebar:
    st.subheader("Search Parameters")
    query = st.text_input("Query", value=' OR '.join([f'"{term}"' for term in KEY_TERMS]), key="query_input")
    default_categories = ["cond-mat.mtrl-sci", "physics.app-ph", "physics.chem-ph", "cond-mat.soft"]
    categories = st.multiselect("Categories", default_categories, default=default_categories, key="categories_select")
    max_results = st.slider("Max Papers", min_value=1, max_value=200, value=10, key="max_results_slider")
    current_year = datetime.now().year
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input("Start Year", min_value=1990, max_value=current_year, value=2010, key="start_year_input")
    with col2:
        end_year = st.number_input("End Year", min_value=start_year, max_value=current_year, value=current_year, key="end_year_input")
    output_formats = st.multiselect("Output Formats", ["SQLite (.db)", "CSV", "JSON"], default=["SQLite (.db)"], key="output_formats_select")
    
    # Cloud-specific settings
    st.subheader("Cloud Settings")
    enable_cloud_limits = st.checkbox("Enable Cloud Optimization", value=os.path.exists("/tmp"), key="cloud_optimization_checkbox")
    max_pdf_downloads = st.slider("Max PDF Downloads", min_value=1, max_value=200, value=10, key="max_pdf_downloads_slider")
    
    search_button = st.button("Search arXiv", key="search_button")
    convert_button = st.button("Update DBs from Existing PDFs", key="convert_button")
    reset_downloads_button = st.button("Reset Downloads", key="reset_downloads_button")

# Reset downloads if button clicked
if reset_downloads_button:
    reset_downloads()
    st.success("Download state reset. You can now perform a new search or download.")

# Main processing
if convert_button:
    if st.session_state.processing:
        st.warning("Processing in progress... Please wait.")
    else:
        st.session_state.processing = True
        with st.spinner("Processing existing PDFs..."):
            batch_convert_pdfs()
        display_logs()
        st.success("DB update complete. Check logs for details.")
        st.session_state.processing = False

# Restore previous search results if available
if st.session_state.search_results and st.session_state.relevant_papers:
    papers = st.session_state.search_results
    relevant_papers = st.session_state.relevant_papers
    df = pd.DataFrame(relevant_papers)
    st.subheader("Papers (Relevance > 30%)")
    st.dataframe(
        df[["id", "title", "year", "categories", "abstract_highlighted", "matched_terms", "relevance_prob", "download_status"]],
        use_container_width=True
    )
    
    # Restore download buttons
    if "SQLite (.db)" in output_formats:
        sqlite_status = save_to_sqlite(df.drop(columns=["abstract_highlighted"]), [])
        st.info(sqlite_status)
    
    if "CSV" in output_formats:
        csv = df.drop(columns=["abstract_highlighted"]).to_csv(index=False)
        st.download_button(
            label="Download Paper Metadata CSV",
            data=csv,
            file_name="seebeck_papers.csv",
            mime="text/csv",
            key="csv_download"
        )
    
    if "JSON" in output_formats:
        json_data = df.drop(columns=["abstract_highlighted"]).to_json(orient="records", lines=True)
        st.download_button(
            label="Download Paper Metadata JSON",
            data=json_data,
            file_name="seebeck_papers.json",
            mime="application/json",
            key="json_download"
        )
    
    # Display individual PDF links
    pdf_paths = st.session_state.download_files.get("pdf_paths", [])
    if pdf_paths:
        st.subheader("Individual PDF Downloads")
        for idx, pdf_path in enumerate(pdf_paths):
            file_data = read_file_for_download(pdf_path)
            if file_data:
                st.download_button(
                    label=f"Download {os.path.basename(pdf_path)}",
                    data=file_data,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                    key=f"pdf_download_{idx}_{time.time()}"
                )
    
        # ZIP download
        zip_path = st.session_state.download_files.get("zip_path")
        if zip_path and os.path.exists(zip_path):
            file_data = read_file_for_download(zip_path)
            if file_data:
                st.download_button(
                    label="Download All PDFs as ZIP",
                    data=file_data,
                    file_name="seebeck_pdfs.zip",
                    mime="application/zip",
                    key=f"zip_download_{time.time()}"
                )
    
    # Database file downloads
    st.subheader("Database Downloads")
    if os.path.exists(METADATA_DB_FILE):
        file_data = read_file_for_download(METADATA_DB_FILE)
        if file_data:
            st.download_button(
                label="Download Metadata Database",
                data=file_data,
                file_name="seebeck_metadata.db",
                mime="application/octet-stream",
                key=f"metadata_db_download_{time.time()}"
            )
    else:
        st.warning(f"Metadata database ({METADATA_DB_FILE}) not found.")
    
    if os.path.exists(UNIVERSE_DB_FILE):
        file_data = read_file_for_download(UNIVERSE_DB_FILE)
        if file_data:
            st.download_button(
                label="Download Universe Database",
                data=file_data,
                file_name="seebeck_universe.db",
                mime="application/octet-stream",
                key=f"universe_db_download_{time.time()}"
            )
    else:
        st.warning(f"Universe database ({UNIVERSE_DB_FILE}) not found.")
    
    display_logs()

if search_button:
    if st.session_state.processing:
        st.warning("Processing in progress... Please wait.")
        st.stop()
        
    if not query.strip():
        st.error("Enter a valid query.")
    elif not categories:
        st.error("Select at least one category.")
    elif start_year > end_year:
        st.error("Start year must be ≤ end year.")
    else:
        st.session_state.processing = True
        st.session_state.download_files = {"pdf_paths": [], "zip_path": None}
        st.session_state.search_params = {
            "query": query,
            "categories": categories,
            "max_results": max_results,
            "start_year": start_year,
            "end_year": end_year
        }
        
        try:
            # Perform health check
            if not system_health_check():
                st.error("System health check failed. Please try again later.")
                reset_processing()
                st.stop()
            
            with st.spinner("Querying arXiv..."):
                papers = query_arxiv(query, categories, max_results, start_year, end_year)
            
            if not papers:
                st.warning("No papers found. Broaden query or categories.")
            else:
                st.success(f"Found **{len(papers)}** papers. Filtering for relevance > 30%...")
                relevant_papers = [p for p in papers if p["relevance_prob"] > 30.0]
                
                if not relevant_papers:
                    st.warning("No papers with relevance > 30%. Broaden query or check logs.")
                else:
                    # Apply cloud limits if enabled
                    if enable_cloud_limits:
                        relevant_papers = limit_pdf_processing(relevant_papers, max_pdfs=max_pdf_downloads)
                    
                    st.success(f"**{len(relevant_papers)}** papers with relevance > 30%. Downloading PDFs...")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    pdf_paths = []
                    
                    for i, paper in enumerate(relevant_papers):
                        status_text.text(f"Downloading {i+1}/{len(relevant_papers)}: {paper['title'][:50]}...")
                        
                        if paper["pdf_url"]:
                            status, pdf_path, content = download_pdf_and_extract(paper["pdf_url"], paper["id"], paper)
                            paper["download_status"] = status
                            paper["pdf_path"] = pdf_path
                            paper["content"] = content
                            if pdf_path:
                                pdf_paths.append(pdf_path)
                        
                        progress_bar.progress((i + 1) / len(relevant_papers))
                        time.sleep(1)  # Avoid rate-limiting
                        update_log(f"Processed paper {i+1}/{len(relevant_papers)}: {paper['title']}")
                        
                        # Clean memory every 3 papers
                        if i % 3 == 0:
                            cleanup_memory()
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Store results in session state
                    st.session_state.search_results = papers
                    st.session_state.relevant_papers = relevant_papers
                    st.session_state.download_files["pdf_paths"] = pdf_paths
                    
                    # Display results
                    df = pd.DataFrame(relevant_papers)
                    st.subheader("Papers (Relevance > 30%)")
                    st.dataframe(
                        df[["id", "title", "year", "categories", "abstract_highlighted", "matched_terms", "relevance_prob", "download_status"]],
                        use_container_width=True
                    )
                    
                    # Export data
                    if "SQLite (.db)" in output_formats:
                        sqlite_status = save_to_sqlite(df.drop(columns=["abstract_highlighted"]), [])
                        st.info(sqlite_status)
                    
                    if "CSV" in output_formats:
                        csv = df.drop(columns=["abstract_highlighted"]).to_csv(index=False)
                        st.download_button(
                            label="Download Paper Metadata CSV",
                            data=csv,
                            file_name="seebeck_papers.csv",
                            mime="text/csv",
                            key=f"csv_download_{time.time()}"
                        )
                    
                    if "JSON" in output_formats:
                        json_data = df.drop(columns=["abstract_highlighted"]).to_json(orient="records", lines=True)
                        st.download_button(
                            label="Download Paper Metadata JSON",
                            data=json_data,
                            file_name="seebeck_papers.json",
                            mime="application/json",
                            key=f"json_download_{time.time()}"
                        )
                    
                    # Display individual PDF links
                    if pdf_paths:
                        st.subheader("Individual PDF Downloads")
                        for idx, pdf_path in enumerate(pdf_paths):
                            file_data = read_file_for_download(pdf_path)
                            if file_data:
                                st.download_button(
                                    label=f"Download {os.path.basename(pdf_path)}",
                                    data=file_data,
                                    file_name=os.path.basename(pdf_path),
                                    mime="application/pdf",
                                    key=f"pdf_download_{idx}_{time.time()}"
                                )
                        
                        # ZIP download
                        zip_path = create_pdf_zip(pdf_paths)
                        if zip_path:
                            st.session_state.download_files["zip_path"] = zip_path
                            file_data = read_file_for_download(zip_path)
                            if file_data:
                                st.download_button(
                                    label="Download All PDFs as ZIP",
                                    data=file_data,
                                    file_name="seebeck_pdfs.zip",
                                    mime="application/zip",
                                    key=f"zip_download_{time.time()}"
                                )
                    
                    # Database file downloads
                    st.subheader("Database Downloads")
                    if os.path.exists(METADATA_DB_FILE):
                        file_data = read_file_for_download(METADATA_DB_FILE)
                        if file_data:
                            st.download_button(
                                label="Download Metadata Database",
                                data=file_data,
                                file_name="seebeck_metadata.db",
                                mime="application/octet-stream",
                                key=f"metadata_db_download_{time.time()}"
                            )
                    else:
                        st.warning(f"Metadata database ({METADATA_DB_FILE}) not found.")
                    
                    if os.path.exists(UNIVERSE_DB_FILE):
                        file_data = read_file_for_download(UNIVERSE_DB_FILE)
                        if file_data:
                            st.download_button(
                                label="Download Universe Database",
                                data=file_data,
                                file_name="seebeck_universe.db",
                                mime="application/octet-stream",
                                key=f"universe_db_download_{time.time()}"
                            )
                    else:
                        st.warning(f"Universe database ({UNIVERSE_DB_FILE}) not found.")
                
                display_logs()
                
        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
            update_log(f"Processing error: {str(e)}")
        finally:
            reset_processing()
            cleanup_memory()

# Display system info in sidebar
with st.sidebar:
    st.subheader("System Info")
    memory_usage = check_memory_usage()
    st.write(f"Memory usage: {memory_usage:.1f} MB")
    st.write(f"Database dir: {DB_DIR}")
    if st.button("Clear Memory Cache", key="clear_memory_cache"):
        cleanup_memory()
        st.success("Memory cache cleared")

# Display logs if available
if st.session_state.log_buffer:
    with st.sidebar:
        st.subheader("Recent Logs")
        for log in list(st.session_state.log_buffer)[-5:]:
            st.text(log)

# Add footer
st.markdown("---")
st.markdown("*Cloud-optimized version - Designed for stable deployment in cloud environments*")
