# 📄 Smart Document Q&A (RAG-based)

A Retrieval-Augmented Generation (RAG) system that lets you upload PDF documents and ask questions about their content in natural language — like having your own personal ChatGPT trained on your documents.

## 🎯 What It Does

Upload one or more PDFs, and ask questions about them. The system retrieves the most relevant sections from your documents and uses an LLM to generate accurate, context-based answers.

## 🛠️ Tech Stack

- **Python** — core language
- **PyPDF2** — PDF text extraction
- **Sentence-Transformers** (`all-MiniLM-L6-v2`) — text embeddings
- **FAISS** — vector similarity search
- **Google Gemini API** — answer generation
- **Streamlit** — web interface

## ⚙️ How It Works

1. **Extract** — Text is extracted from uploaded PDF(s)
2. **Chunk** — Text is split into overlapping chunks (500 characters, 50 character overlap) to preserve context
3. **Embed** — Each chunk is converted into a 384-dimensional vector using a sentence-transformer model
4. **Store** — Vectors are stored in a FAISS index for fast similarity search
5. **Retrieve** — When a question is asked, its embedding is compared against stored vectors to find the most relevant chunks
6. **Generate** — The relevant chunks + question are sent to Gemini, which generates a natural-language answer grounded in the document content

## 🚀 Features

- Supports multiple PDF uploads at once — the system searches across all documents to answer questions
- Handles edge cases gracefully (e.g., scanned PDFs with no extractable text)
- Session-aware — automatically reprocesses documents only when the uploaded files change

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🔑 Setup

1. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a `.env` file in the project root:
GEMINI_API_KEY=your_key_here
## ▶️ Usage

**Web interface (recommended):**
```bash
streamlit run app.py
```

**Command-line version:**
```bash
python main.py
```

## 📁 Project Structure
├── read_pdf.py # PDF text extraction
├── chunking.py # Text chunking logic
├── embeddings.py # Embedding generation
├── vector_store.py # FAISS index + similarity search
├── main.py # CLI version + LLM integration
├── app.py # Streamlit web interface
└── requirements.txt # Dependencies
## 💡 Key Learnings

- Implemented the full RAG pipeline from scratch — no pre-built RAG libraries used
- Debugged and fixed a session-state caching bug where stale document data persisted across new uploads
- Identified and addressed a retrieval limitation with multi-topic queries by tuning the `top_k` parameter