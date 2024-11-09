import os
import streamlit as st
import logging
import tempfile  # For creating temporary directories
import shutil    # For file and directory operations
# from sentence_transformers import SentenceTransformer  # Uncomment if you need SentenceTransformer directly
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from typing import Optional, List
from langchain.schema import Document

# Configure logging settings to track application events and errors
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)  # Initialize logger for this module


def create_vector_db_from_text(text_list: List[str]) -> Optional[Chroma]:
    """
    Create a vector database from a list of text strings using SentenceTransformer embeddings.
    
    Args:
        text_list (List[str]): A list of text strings to be processed.

    Returns:
        Optional[Chroma]: A vector store containing the processed document chunks, or None if creation fails.
    """
    try:
        logger.info("Creating vector DB from list of text strings")

        # Filter out any None or empty strings from text_list
        valid_texts = [text for text in text_list if text]

        # Check for valid content in the text list
        if not valid_texts:
            logger.error("Text list is empty or contains only invalid entries.")
            return None

        # Convert each valid text string into a Document object
        data = [Document(page_content=text) for text in valid_texts]

        # Split the document text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        logger.info("Document split into chunks")

        # Initialize embeddings using the HuggingFace model
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Set the directory where the vector DB will be persisted
        persist_directory = 'db'

        # Create a Chroma vector DB from the document chunks
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=embedding, 
            persist_directory=persist_directory
        )
        logger.info("Vector DB created and persisted")

        return vector_db

    except Exception as e:
        logger.error(f"Error creating vector database: {e}")
        return None

def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.
    
    Args:
        vector_db (Optional[Chroma]): The vector database to be deleted.
    """
    logger.info("Deleting vector DB")

    # Check if the vector DB exists before attempting deletion
    if vector_db is not None:
        # Delete the vector DB and clear related session state variables
        vector_db.delete_collection()
        st.session_state.pop("pdf_pages", None)
        st.session_state.pop("file_upload", None)
        st.session_state.pop("vector_db", None)
        st.success("Collection and temporary files deleted successfully.")
        logger.info("Vector DB and related session state cleared")

        # Rerun the Streamlit app to reflect the changes
        st.rerun()
    else:
        # Display an error if no vector DB is found to delete
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")
