from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH) # Load all PDFs from the data directory
    documents = loader.load() # Load the documents in the form of Document objects. Example: [Document(page_content="..."), Document(page_content="...")]
    print(f"Loaded {len(documents)} pages from PDFs.") # Print the number of pages loaded
    return documents # Return the loaded documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter( # Split the text into chunks
        chunk_size=500, # Maximum number of characters per chunk
        chunk_overlap=100, # Number of characters to overlap between chunks
        length_function=len, # Function to calculate the length of each chunk
        add_start_index=True, # Add the start index of each chunk
    )
    chunks = text_splitter.split_documents(documents) # Split the documents into chunks. Example: [Document(page_content="..."), Document(page_content="...")]
    print(f"Split {len(documents)} pages into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH): # it checks if a chroma folder already exists. If it does, it deletes it completely and starts fresh. This prevents old data from mixing with new data every time you run the file.
        shutil.rmtree(CHROMA_PATH) # Remove the existing Chroma directory
    
    embedding_function = HuggingFaceEmbeddings( # Initialize the embedding function
        model_name="all-MiniLM-L6-v2"
    )
    
    db = Chroma.from_documents( # Create the Chroma database
        chunks,
        embedding_function,
        persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to ChromaDB.")

if __name__ == "__main__":
    main()