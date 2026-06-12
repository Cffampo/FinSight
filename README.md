# 📈 FinSight

An AI-powered financial document assistant that lets you ask natural language questions about SEC filings using Retrieval-Augmented Generation (RAG).

Built with Python, LangChain, ChromaDB, and the Anthropic API (Claude).

---

## What It Does

FinSight allows you to upload SEC filings (10-K annual reports) and ask specific financial questions about them. Instead of manually searching through hundreds of pages, you can ask questions like:

- *"What were Apple's total net sales in fiscal year 2025?"*
- *"What risks did JPMorgan mention in their most recent annual report?"*
- *"How did Morningstar's operating income change year over year?"*

The app retrieves the most relevant sections from the documents and uses Claude to generate accurate, sourced answers.

---

## Tech Stack

- **Python** — core application logic
- **LangChain** — document loading, text splitting, and RAG orchestration
- **ChromaDB** — vector database for semantic search
- **HuggingFace Embeddings** — converts text chunks into vectors (`all-MiniLM-L6-v2`)
- **Anthropic API (Claude)** — generates answers from retrieved context
- **Streamlit** — frontend UI

---

## How It Works

1. SEC filings (PDFs) are loaded and split into chunks using LangChain
2. Each chunk is converted into a vector using a HuggingFace embedding model
3. Vectors are stored in ChromaDB for semantic search
4. When a user asks a question, it is converted into a vector and matched against the stored chunks
5. The most relevant chunks are retrieved and sent to Claude as context
6. Claude generates an accurate answer grounded in the source documents

---

## Current Data Sources

- Apple 10-K (Fiscal Year 2025)
- JPMorgan Chase 10-K (Fiscal Year 2024)
- Morningstar 10-K (Fiscal Year 2024)

---

## Getting Started

### Prerequisites
- Python 3.9+
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com))

### Installation

```bash
# Clone the repository
git clone https://github.com/Cffampo/FinSight.git
cd FinSight

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install anthropic langchain langchain-community langchain-chroma langchain-huggingface langchain-text-splitters chromadb pypdf python-dotenv sentence-transformers streamlit
```

### Setup

Create a `.env` file in the root directory:

```
ANTHROPIC_API_KEY=your-api-key-here
```

Add your SEC filing PDFs to the `data/` folder, then build the vector database:

```bash
python create_database.py
```

### Run the App

```bash
streamlit run app.py
```

---

## Project Structure

```
FinSight/
├── data/                  # SEC filing PDFs
├── chroma/                # ChromaDB vector store (auto-generated)
├── create_database.py     # Loads PDFs, chunks text, builds ChromaDB
├── query_data.py          # RAG query logic and Claude integration
├── app.py                 # Streamlit frontend
├── .env                   # API keys (never committed)
└── .gitignore
```

---

## Author

Carl Fampo — [LinkedIn](https://linkedin.com/in/carl-fampo) | [GitHub](https://github.com/Cffampo)
