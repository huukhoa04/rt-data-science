from core.pinecone_setup import pc
from core.config import RECORD_NAMESPACE, MOVIES_HOST, PINECONE_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM
import httpx
from typing import List, Dict, Any, Optional
import asyncio
import json

class VectorService:
    def __init__(self):
        self.index = pc
        self.api_key = PINECONE_API_KEY
        self.embedding_model = EMBEDDING_MODEL
        self.embedding_dim = EMBEDDING_DIM

    # Pinecone Embedding Inference Service
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text using Pinecone's inference API.
        
        Args:
            texts: List of text strings to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        # Format the input according to Pinecone API requirements
        inputs = [{"text": text} for text in texts]
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.pinecone.io/embed",
                headers={
                    "Api-Key": self.api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-Pinecone-API-Version": "2025-04"
                },
                json={
                    "model": self.embedding_model,
                    "inputs": inputs,
                    "parameters": {
                        "input_type": "passage",
                        "dimension": self.embedding_dim,
                        "truncate": "END"
                    }
                },
                timeout=60.0  # Increased timeout for larger batches
            )
            
            if response.status_code != 200:
                raise Exception(f"Error from Pinecone Embedding API: {response.text}")
                
            result = response.json()
            embeddings = result.get("data", [])

            # Validate the dimensions of returned embeddings
            for embedding in embeddings:
                if len(embedding.get('values')) != self.embedding_dim:
                    raise ValueError(
                        f"Expected embedding dimension {self.embedding_dim}, "
                        f"but got {len(embedding.get('values'))} from Pinecone API"
                    )
                    
            return embeddings

    # Helper method to get single text embedding
    async def get_text_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text input.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            List of float values representing the embedding vector
        """
        try:
            embeddings = await self.generate_embeddings([text])
            
            # Debug information
            if embeddings and len(embeddings) > 0:
                print(f"Generated embedding with length: {len(embeddings[0].get('values'))}")
            else:
                print("No embeddings generated")
                
            return embeddings[0].get('values') if embeddings else []
        except Exception as e:
            print(f"Error generating text embedding: {str(e)}")
            # Return a zero vector of the expected dimension as fallback
            return [0.0] * self.embedding_dim

    # Modified search_text method to use our embedding service
    async def search_text(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search vectors using text input via embeddings.
        
        Args:
            text: The text query
            top_k: Maximum number of results to return
            
        Returns:
            Dictionary containing the search results
        """
        # Generate embedding for the text query
        embedding = await self.get_text_embedding(text)
        
        if not embedding:
            raise ValueError("Failed to generate embedding for the text query")
        
        # Use the embedding to perform vector search
        return await self.search_vector(vector=embedding, top_k=top_k)

    # querying vectors
    async def query_vectors(self, vector, top_k=5):
        return self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True
        )
        
    async def upsert_vectors(self, vectors):
        return self.index.upsert(vectors=vectors)
    
    async def list_vectors(self, namespace=RECORD_NAMESPACE, pagination_token=None, limit=100):
        """
        List vector IDs with pagination support.
        
        Args:
            namespace: The namespace to list vectors from
            pagination_token: Token for pagination, None for first page
            limit: Maximum number of results to return
            
        Returns:
            Dictionary containing vector IDs and pagination token for next page
        """
        try:
            # Ensure we have a manageable limit to prevent URI issues
            safe_limit = min(limit, 50)
            
            response = self.index.list_paginated(
                namespace=namespace,
                pagination_token=pagination_token,
                limit=safe_limit
            )
            # Extract vector IDs from the response - each item is a dict with 'id' key
            vector_ids = [vector['id'] for vector in response.get('vectors', [])]
            next_token = response.get('pagination', {}).get('next')
            
            return {
                "vector_ids": vector_ids,
                "next_pagination_token": next_token
            }


        except Exception as e:
            print(f"Error in list_vectors: {str(e)}")
            # Return empty result on error
            return {
                "vector_ids": [],
                "next_pagination_token": None
            }
    async def fetch_vectors(self, ids, namespace=RECORD_NAMESPACE):
        """
        Fetch vectors by their IDs using direct HTTP request.
        
        Args:
            ids: List of vector IDs to fetch
            namespace: The namespace to fetch vectors from
            
        Returns:
            Dictionary containing the requested vectors and their metadata
        """
        try:
            # Handle empty ids list
            if not ids:
                return {}
                
            # Build the URL with query parameters
            url = f"{MOVIES_HOST}/vectors/fetch"
            
            # Prepare query parameters - multiple ids are passed as repeated query params
            params = []
            for id in ids:
                params.append(("ids", id))
                
            if namespace:
                params.append(("namespace", namespace))
                
            # Make the HTTP request
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    params=params,
                    headers={
                        "Api-Key": self.api_key,
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "X-Pinecone-API-Version": "2025-04"
                    },
                    timeout=60.0
                )
                
                if response.status_code != 200:
                    print(f"Error fetching vectors: Status {response.status_code}, Response: {response.text}")
                    return {}
                    
                # Parse the response
                result = response.json()
                print(f"Fetched vectors: {result}")
                return result.get("vectors", {})
                
        except Exception as e:
            print(f"Error in fetch_vectors: {str(e)}")
            return {}

    # For suggestion query and semantic search
    async def search_vector(self, vector, top_k=5):
        return self.index.query(
            namespace=RECORD_NAMESPACE,
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            include_values=True,
        )
    
    # For filtering search results based on metadata fields
    async def search_filtered(self, text: str, filter_fields: List[str], top_k: int = 10) -> Dict[str, Any]:
        """
        Search vectors with text filtering on specific metadata fields.
        
        Args:
            text: The search query text
            filter_fields: List of metadata fields to search in (e.g., ["title", "genres"])
            top_k: Maximum number of results to return
            
        Returns:
            Dictionary containing the search results
        """

        # Generate embedding for the text query
        embedding = await self.get_text_embedding(text)
        
        if not embedding:
            raise ValueError("Failed to generate embedding for the text query")
        
        # Prepare filter conditions for each field
        # This implementation will use a simple string search
        # For each filter field, we check if the query appears in that field
        filter_conditions = {}
        query_lower = text.lower()
        
        for field in filter_fields:
            # For each field, create a filter condition that checks if the text is contained in the field
            filter_conditions[field] = {"$text": {"$contains": query_lower}}
        
        # Build the combined filter query (match any of the conditions)
        filter_query = {"$or": []}
        for field, condition in filter_conditions.items():
            field_query = {f"metadata.{field}": condition}
            filter_query["$or"].append(field_query)
        
        # Return the query results
        return self.index.query(
            namespace=RECORD_NAMESPACE,
            vector=embedding,
            top_k=top_k,
            filter=filter_query,
            include_metadata=True,
            include_values=True
        )