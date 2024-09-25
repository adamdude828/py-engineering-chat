import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from py_engineering_chat.util.chat_settings_manager import ChatSettingsManager
from py_engineering_chat.util.logger_util import get_configured_logger

class ChromaDB:
    def __init__(self):
        self.settings_manager = ChatSettingsManager()
        self.logger = get_configured_logger(__name__)
        self.client = self._initialize_client()
        self.collection = self._get_or_create_collection()

    def _initialize_client(self):
        ai_shadow_directory = self.settings_manager.get_ai_shadow_directory()
        chroma_db_path = os.path.join(ai_shadow_directory, '.chroma_db')
        self.logger.debug(f"Initializing Chroma client with path: {chroma_db_path}")
        return chromadb.PersistentClient(path=chroma_db_path)

    def _get_or_create_collection(self):
        collection_name = "conversation_history"
        try:
            return self.client.get_collection(name=collection_name)
        except ValueError:
            self.logger.info(f"Creating new collection: {collection_name}")
            return self.client.create_collection(name=collection_name)

    def add_conversation(self, conversation_id: str, content: str, metadata: Dict[str, Any], embedding: List[float]):
        self.logger.debug(f"Adding conversation with ID: {conversation_id}")
        self.collection.add(
            ids=[conversation_id],
            documents=[content],
            metadatas=[metadata],
            embeddings=[embedding]
        )

    def get_conversation(self, conversation_id: str):
        self.logger.debug(f"Retrieving conversation with ID: {conversation_id}")
        result = self.collection.get(ids=[conversation_id], include=['embeddings', 'documents', 'metadatas'])
        if result and result['documents']:
            return {
                'id': conversation_id,
                'content': result['documents'][0],
                'metadata': result['metadatas'][0],
                'embedding': result['embeddings'][0]
            }
        return None

    def update_conversation(self, conversation_id: str, content: str, metadata: Dict[str, Any], embedding: List[float]):
        self.logger.debug(f"Updating conversation with ID: {conversation_id}")
        self.collection.update(
            ids=[conversation_id],
            documents=[content],
            metadatas=[metadata],
            embeddings=[embedding]
        )

    def delete_conversation(self, conversation_id: str):
        self.logger.debug(f"Deleting conversation with ID: {conversation_id}")
        self.collection.delete(ids=[conversation_id])

    def search_conversations(self, query_embedding: List[float], n_results: int = 5):
        self.logger.debug(f"Searching conversations with {n_results} results")
        total_elements = self.collection.count()
        if total_elements == 0:
            self.logger.warning("No elements in the collection. Returning empty result.")
            return []
        
        if n_results > total_elements:
            self.logger.warning(f"Requested {n_results} results, but only {total_elements} elements exist. Adjusting n_results.")
            n_results = total_elements
        
        n_results = max(1, n_results)  # Ensure n_results is at least 1
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return [
            {
                'id': id_,
                'content': document,
                'metadata': metadata,
                'distance': distance
            }
            for id_, document, metadata, distance in zip(
                results['ids'][0],
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )
        ]

    def get_conversations_by_metadata(self, metadata_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        self.logger.debug(f"Retrieving conversations with metadata filter: {metadata_filter}")
        results = self.collection.get(where=metadata_filter, include=['embeddings', 'documents', 'metadatas'])
        
        if not results['documents']:
            self.logger.debug("No conversations found matching the metadata filter")
            return []
        
        return [
            {
                'id': id_,
                'content': document,
                'metadata': metadata,
                'embedding': embedding
            }
            for id_, document, metadata, embedding in zip(
                results['ids'],
                results['documents'],
                results['metadatas'],
                results['embeddings']
            )
        ]
