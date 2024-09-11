from sentence_transformers import SentenceTransformer
import chromadb
from .logger_util import get_configured_logger
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path='../.env')

# Get a logger instance
logger = get_configured_logger(__name__)

def search_chroma(collection_name: str, query: str) -> list:
    try:
        logger.debug(f"Starting search in collection: {collection_name} with query: {query}")
        
        # Get AI_SHADOW_DIRECTORY from environment variables
        ai_shadow_directory = os.getenv('AI_SHADOW_DIRECTORY')
        if not ai_shadow_directory:
            raise ValueError("AI_SHADOW_DIRECTORY environment variable is not set")

        # Construct the Chroma DB path
        chroma_db_path = os.path.join(ai_shadow_directory, '.chroma_db')
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        client = chromadb.PersistentClient(path=chroma_db_path)
        collection = client.get_collection(name=collection_name)
        
        query_embedding = model.encode([query]).tolist()
        
        results = collection.query(query_embeddings=query_embedding, n_results=3)
        logger.debug(f"Search results: {results['documents']}")
        
        return results['documents']
    except Exception as e:
        logger.error(f"Error searching context: {str(e)}")
        return []