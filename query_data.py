from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a financial analyst assistant. Answer the question based only on the following context from SEC filings:

{context}

---

Answer this question based on the context above: {question}

If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided documents to answer that."
"""

def query_rag(question: str):
    # Load the embedding function
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Load ChromaDB
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    # Search for relevant chunks
    results = db.similarity_search_with_relevance_scores(question, k=10) # Search for the most relevant chunks

    if len(results) == 0 or results[0][1] < 0.1: # If no relevant results found or the most relevant result has a low relevance score
        print("Unable to find relevant results.")
        return

    # Build context from retrieved chunks
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results]) # Join the content of the retrieved chunks with a separator

    # Format the prompt
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=question) # Format the prompt with the context and question

    # Send to Claude
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) # Initialize the Anthropic client with the API key
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    response = message.content[0].text

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    
    return {
        "answer": response,
        "sources": sources
    }