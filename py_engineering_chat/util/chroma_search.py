from sentence_transformers import SentenceTransformer
import chromadb

def search_chroma(collection_name: str, query: str) -> List[str]:
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        client = chromadb.PersistentClient(path='.chroma_db')
        collection = client.get_collection(name=collection_name)
        query_embedding = model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=3)
        return results['documents'][0]
    except Exception as e:
        print(f"Error searching context: {str(e)}")
        return []