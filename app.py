from __future__ import annotations

import streamlit as st
from nlp_engine import SemanticSearchEngine, DocumentProcessingError

st.set_page_config(
    page_title="CampusFind | Semantic Document Search",
    page_icon="🔎",
    layout="wide",
)

st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1150px;}
    .hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(120deg, #172554, #1e40af); color: white; margin-bottom: 1.2rem;}
    .hero h1 {margin: 0; font-size: 2.1rem; color: white;}
    .hero p {margin: .45rem 0 0 0; color: #dbeafe;}
    .result-card {border: 1px solid #dbe3ef; border-radius: 13px; padding: 1rem 1.1rem; margin: .7rem 0; background: var(--secondary-background-color);}
    .source-line {font-size: .9rem; color: #64748b;}
    .score {font-weight: 700; color: #2563eb;}
</style>
<div class="hero">
  <h1> CampusFind</h1>
  <p>Semantic Search Engine for College Documents · Find relevant passages by meaning, not just exact keywords.</p>
</div>
""", unsafe_allow_html=True)

if "engine" not in st.session_state:
    st.session_state.engine = SemanticSearchEngine()
if "indexed_names" not in st.session_state:
    st.session_state.indexed_names = []

engine: SemanticSearchEngine = st.session_state.engine

with st.sidebar:
    st.header("📄 Your document collection")
    uploads = st.file_uploader(
        "Upload college PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more text-based PDF files. Scanned/image-only PDFs need OCR and are not supported by this demo.",
    )
    chunk_size = st.number_input("Chunk size (words)", min_value=80, max_value=300, value=150, step=10)
    overlap = st.number_input("Overlap (words)", min_value=0, max_value=80, value=30, step=5)
    result_count = st.slider("Number of results", min_value=1, max_value=10, value=5)
    index_clicked = st.button("Build / replace search index", type="primary", use_container_width=True)
    clear_clicked = st.button("Clear indexed documents", use_container_width=True)
    st.caption("Files stay in this running app session; they are not uploaded to an external service.")

if clear_clicked:
    engine.clear()
    st.session_state.indexed_names = []
    st.success("Search index cleared.")

if index_clicked:
    if not uploads:
        st.warning("Upload at least one PDF before building the index.")
    else:
        docs = [(item.name, item.getvalue()) for item in uploads]
        with st.spinner("Reading PDFs and creating semantic embeddings… The first run may download the model."):
            try:
                count = engine.add_pdfs(docs, chunk_size=int(chunk_size), overlap=int(overlap))
                st.session_state.indexed_names = [name for name, _ in docs]
                if count:
                    st.success(f"Index ready: {count} text chunks from {len(docs)} PDF(s).")
                else:
                    st.warning("No searchable text was found. These may be scanned PDFs; OCR is not included.")
            except DocumentProcessingError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not build the index: {exc}")

col_a, col_b, col_c = st.columns(3)
col_a.metric("Indexed chunks", engine.chunk_count)
col_b.metric("PDF files", len(engine.source_names))
col_c.metric("Embedding model", "MiniLM-L6-v2")

if engine.source_names:
    with st.expander("Currently indexed files", expanded=False):
        for name in engine.source_names:
            st.write(f"📄 {name}")

st.subheader("Ask a question")
examples = [
    "What are the attendance requirements?",
    "How can a student apply for leave?",
    "What is the process for semester examinations?",
    "What facilities are available in the hostel?",
    "Where can students find placement information?",
]
selected_example = st.selectbox("Try an example question", ["Choose an example…"] + examples)
default_query = "" if selected_example == "Choose an example…" else selected_example
query = st.text_input(
    "Search in natural language",
    value=default_query,
    placeholder="e.g., What should I do if I miss an examination?",
)
search_clicked = st.button("Search documents", type="primary")

if search_clicked:
    if not engine.chunk_count:
        st.warning("Upload PDFs and build the index first.")
    elif not query.strip():
        st.warning("Enter a question to search.")
    else:
        with st.spinner("Finding the most relevant passages…"):
            try:
                results = engine.search(query.strip(), top_k=int(result_count))
            except Exception as exc:
                st.error(f"Search failed: {exc}")
                results = []
        if results:
            st.markdown(f"### Results for: *{query.strip()}*")
            st.caption("Similarity scores indicate how close the query and passage embeddings are. They are ranking signals, not probabilities of correctness.")
            for rank, result in enumerate(results, start=1):
                score = result["score"]
                st.markdown(
                    f"""<div class="result-card">
                    <strong>{rank}. {result['source']}</strong>
                    <div class="source-line">Page {result['page']} · <span class="score">Similarity: {score:.3f}</span></div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                with st.expander(f"View passage {rank}", expanded=(rank == 1)):
                    st.write(result["text"])
        else:
            st.info("No matching passages were found. Try a different wording or upload documents with searchable text.")

st.divider()
st.caption("Demo project · Python + Streamlit + Sentence Transformers + FAISS. Verify important policies against official college documents.")
