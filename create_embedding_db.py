"""
Create an embedding database from all the marine heatwave discussions.
"""


import logging
import os
import shutil
from datetime import datetime
from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# set up logging and formatting of timestamp
def setup_logging():
    """Setup logging to both console and file"""
    # Create logs directory if it doesn't exist
    from pathlib import Path

    # create logs directory if not exists
    Path('logs').mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger(__name__)
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
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (for saving to log file)
    script_name = Path(__file__).stem  # Gets filename without .py extension
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/{script_name}_{timestamp}.log'
    
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized - console + file: {log_filename}")
    return logger


# Initialize logging
logger = setup_logging()

def load_documents() -> list[Document]:
    """Load markdown documents from the data directory.
    Using the DirectoryLoader from langchain_community.
    https://python.langchain.com/docs/how_to/document_loader_directory/

    Returns
    -------
    list
        A list of loaded documents.
    """
    logger.info("Loading documents from data/ directory...")
    loader = DirectoryLoader('data/', glob='*.md')
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents")
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
    logger.info(f"Chunking {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks

def create_embedding_db(chunks: list[Document], LLM_EMBEDDING_MODEL: str, CHROMA_PATH: str='chroma_db'):
    """Create an embedding database from the document chunks.
    Using the Chroma vector store from langchain.
    https://python.langchain.com/docs/integrations/text_embedding/

    Parameters
    ----------
    chunks : list
        A list of document chunks.
    """
    logger.info(f"Creating embedding database at {CHROMA_PATH}...")

    # clean up the database if exists
    if os.path.exists(CHROMA_PATH):
        logger.info(f"Existing database found at {CHROMA_PATH}, cleaning up...")
        # Note: Need to delete before creating to avoid conflicts
        shutil.rmtree(CHROMA_PATH)
        logger.info(f"Deleted existing database at {CHROMA_PATH}")

    logger.info("Initializing embeddings model...")
    embeddings = OllamaEmbeddings(model=LLM_EMBEDDING_MODEL)

    logger.info(f"Creating Chroma database with {len(chunks)} chunks...")
    db = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name="marine_heatwave_discussions",
        persist_directory=CHROMA_PATH)

    # Note: persist() method is no longer needed in newer langchain-chroma versions
    # The database is automatically persisted when persist_directory is specified
    logger.info(f"Embedding database created and persisted at {CHROMA_PATH}")

def main(CHROMA_PATH: str,LLM_EMBEDDING_MODEL:str):
    logger.info("Starting marine heatwave embedding database creation...")

    # load the documents
    documents = load_documents()

    # split the documents into chunks
    chunks = document_chunking(documents)

    # create the embedding database
    create_embedding_db(chunks, LLM_EMBEDDING_MODEL, CHROMA_PATH)

    logger.info("Marine heatwave embedding database creation completed successfully!")

if __name__ == "__main__":
    # define the chroma database path
    CHROMA_PATH = 'chroma_db'
    LLM_EMBEDDING_MODEL = 'nomic-embed-text'
    main(CHROMA_PATH,LLM_EMBEDDING_MODEL)
