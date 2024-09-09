import chromadb

def list_collections():
    """List available collections in Chroma."""
    client = chromadb.PersistentClient(path="./.chroma_db")
    collections = client.list_collections()
    if collections:
        print("Available collections in Chroma:")
        for collection in collections:
            print(f"- Name: {collection.name}")
            print(f"  Number of documents: {collection.count()}")
            metadata = collection.metadata
            if metadata:
                print(f"  Metadata: {metadata}")
            print()  # Add a blank line between collections for better readability
    else:
        print("No collections found in Chroma.")

import json
import uuid
import os

def list_collection_content(collection_name):
    """List content of a specific collection in Chroma and save to a JSON file."""
    client = chromadb.PersistentClient(path="./.chroma_db")
    try:
        collection = client.get_collection(name=collection_name)
        results = collection.get()
        
        data = {
            "collection_name": collection_name,
            "items": [
                {
                    "id": id,
                    "document": document,
                    "metadata": metadata
                }
                for id, document, metadata in zip(results['ids'], results['documents'], results['metadatas'])
            ],
            "total_items": len(results['ids'])
        }
        
        filename = f"collection_content_{uuid.uuid4().hex[:8]}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Collection content saved to file: {filename}")
        print(f"Total items in collection: {data['total_items']}")
    except ValueError:
        print(f"Collection '{collection_name}' not found.")