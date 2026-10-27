# CampusFind — Semantic Search Engine for College Documents

A small NLP project that retrieves relevant passages from PDF documents using sentence embeddings and vector similarity.

## Features

- Upload multiple text-based PDFs.
- Extract text page by page with PyMuPDF.
- Split text into overlapping word chunks while retaining source filename and page number.
- Encode chunks with `sentence-transformers/all-MiniLM-L6-v2`.
- Normalize embeddings and search with FAISS inner product (equivalent to cosine similarity for normalized vectors).
- Show ranked passages, source, page, and similarity score.

## Requirements

- Python 3.10–3.12 recommended.
- Internet access for the first model download.
- Text-based PDFs. Scanned/image-only PDFs require OCR, which this demo does not include.

## Run on Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.\.venv\Scripts\python.exe -m streamlit run app.py`.

## Run on Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Demo steps

1. Start the app.
2. In the sidebar, upload two or more college-related PDFs.
3. Click **Build / replace search index** and wait for the model and index to load.
4. Ask a question in your own words or choose an example.
5. Compare the source, page, passage, and similarity scores of the top results.

Example queries:
- What are the attendance requirements?
- How can a student apply for leave?
- What is the process for semester examinations?
- What facilities are available in the hostel?
- Where can students find placement information?

Use real college documents for meaningful results. Example questions are generic and do not imply any particular policy exists in your uploaded files.

## How it works

1. **Text extraction:** PyMuPDF extracts text from each PDF page.
2. **Chunking:** Page text is split into chunks of 150 words with 30-word overlap by default. Overlap helps preserve context across chunk boundaries.
3. **Embeddings:** Sentence Transformers maps each chunk to a dense numerical vector.
4. **Indexing:** FAISS stores the vectors in an exact inner-product index.
5. **Retrieval:** The query is embedded in the same vector space. Both query and document vectors are normalized, so inner product gives cosine similarity. The highest-scoring passages are shown first.

## Limitations

- Scanned PDFs and image-only pages are not OCR-processed.
- Results depend on the quality and content of the uploaded documents.
- Similarity scores are ranking signals, not probabilities or proof that a passage is correct.
- This is a local academic demo, not an official college policy service.

## Project structure

```text
college-semantic-search/
├── app.py
├── nlp_engine.py
├── requirements.txt
├── README.md
├── .gitignore
└── sample_documents/
    └── demo_college_handbook.pdf
```
