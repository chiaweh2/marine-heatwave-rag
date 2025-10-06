"""
Create an embedding database from all the marine heatwave discussions.

Note: need to improve if additional discussions are added in the future.
try avoid re-creating the entire database if only a few new documents are added.
vector ids should be added accordingly so when adding new documents,
the existing vectors/embeddings do not need to be re-created.
"""

import logging
import os
from pathlib import Path
import shutil
from datetime import datetime
from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def setup_logging(debug: bool=False) -> logging.Logger:
    """Setup logging to both console and file
    Create a logs folder if not exists.
    Create a log file with the script name and timestamp in the logs folder.

    Parameters
    ----------
    debug : bool, optional
        Whether to enable debug logging, by default False
    Returns
    -------
    logging.Logger
        The configured logger instance.
    """

    # create logs directory if not exists
    os.makedirs('logs', exist_ok=True)

    # Create logger
    logger = logging.getLogger(__name__)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (for terminal output)
    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (for saving to log file)
    script_name = Path(__file__).stem  # Gets filename without .py extension
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/{script_name}_{timestamp}.log'

    file_handler = logging.FileHandler(log_filename)
    if debug:
        file_handler.setLevel(logging.DEBUG)
    else:
        file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def load_documents() -> list[Document]:
    """Load markdown documents from the data directory.
    Using the DirectoryLoader from langchain_community.
    https://python.langchain.com/docs/how_to/document_loader_directory/

    Returns
    -------
    list
        A list of loaded documents.
    """
    loader = DirectoryLoader('data/', glob='*.md')
    documents = loader.load()
    return documents

def document_chunking(documents: list[Document]) -> list[Document]:
    """Chunk the documents into smaller pieces for easier processing.
    Usually the discussion are relatively short for about ~2500 characters.
    Choose the chunk size and chunk overlap based on the average length of the documents.
    Here we use a chunk size of 1000 characters with an overlap of 100 characters

    Parameters
    ----------
    documents : list
        A list of documents to chunk.

    Returns
    -------
    list
        A list of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def embedding_model():
    """Initialize the embedding model.
    This function should be used for creating the 
    embedding vector database and also used for 
    creating the query embedding in rag_prompt.py

    Returns
    -------
    OllamaEmbeddings
        The initialized embedding model.
    """
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    return embeddings

def create_embedding_db(
    chunks: list[Document],
    chroma_path: str='chroma_db'
) -> None:
    """Create an embedding vectorstore database from the document chunks.
    Using the Chroma vector store from langchain.
    https://python.langchain.com/docs/integrations/text_embedding/

    Parameters
    ----------
    chunks : list
        A list of document chunks.
    chroma_path : str, optional
        Path to save the Chroma embedding database, by default 'chroma_db'

    Returns
    -------
    None

    """
    embeddings = embedding_model()

    # Note: persist() method is no longer needed in newer langchain-chroma versions
    # The database is automatically persisted when persist_directory is specified
    _ = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name="marine_heatwave_discussions",
        persist_directory=chroma_path)

def main(chroma_path: str):
    """
    Main function to create marine heatwave embedding database.
    """
    # Initialize logging
    logger_embedding = setup_logging()
    logger_embedding.info("Starting marine heatwave embedding database creation...")

    # Load the documents
    logger_embedding.info("Loading documents from data/ directory...")
    documents = load_documents()
    logger_embedding.info("Loaded %d documents", len(documents))

    # Split the documents into chunks
    logger_embedding.info("Chunking %d documents...", len(documents))
    chunks = document_chunking(documents)
    logger_embedding.info("Created %d chunks from %d documents", len(chunks), len(documents))

    # Create the embedding database
    logger_embedding.info("Creating embedding database at %s...", chroma_path)
    if os.path.exists(chroma_path):
        logger_embedding.info("Existing database found at %s, cleaning up...", chroma_path)
        shutil.rmtree(chroma_path)
        logger_embedding.info("Deleted existing database at %s", chroma_path)
    logger_embedding.info("Initializing embeddings model...")
    logger_embedding.info("Creating Chroma database with %d chunks...", len(chunks))

    create_embedding_db(chunks, chroma_path)

    logger_embedding.info("Embedding database created and persisted at %s", chroma_path)
    logger_embedding.info("Marine heatwave embedding database creation completed successfully!")

if __name__ == "__main__":
    # define the chroma database path
    CHROMA_PATH = 'chroma_db'
    main(CHROMA_PATH)
