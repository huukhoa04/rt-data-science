from pinecone import Pinecone
from core.config import PINECONE_API_KEY, MOVIES_HOST, INDEX_NAME, RECORD_NAMESPACE


pc = Pinecone(
    api_key=PINECONE_API_KEY).Index(name=INDEX_NAME, host=MOVIES_HOST)
    
# The above code initializes a Pinecone client using the provided API key and host, and sets up an index for querying.

if __name__ == "__main__":
    # Example usage of the Pinecone client
    try:
        # Check if the index exists
        result = pc.list_paginated(
                namespace=RECORD_NAMESPACE, 
                limit=10,
                pagination_token=None
            )
        print(f"Index exists: {result}")
    except Exception as e:
        print(f"Error checking index: {e}")