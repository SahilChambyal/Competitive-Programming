"""Vector store management using Pinecone"""

import pinecone
from typing import List, Dict, Any, Optional
from ..config import settings


class VectorStore:
    """Manages vector storage and retrieval using Pinecone"""
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        
        # Initialize Pinecone
        pinecone.init(
            api_key=api_key,
            environment=environment
        )
        
        # Create or connect to index
        self.index = self._get_or_create_index()
    
    def _get_or_create_index(self):
        """Get existing index or create new one"""
        if self.index_name not in pinecone.list_indexes():
            print(f"Creating new Pinecone index: {self.index_name}")
            pinecone.create_index(
                name=self.index_name,
                dimension=settings.embedding_dimension,
                metric="cosine"
            )
        
        return pinecone.Index(self.index_name)
    
    def upsert_documents(self, embedded_docs: List[Dict[str, Any]]) -> bool:
        """Upload embedded documents to vector store"""
        try:
            vectors = []
            for doc in embedded_docs:
                vector = {
                    "id": doc["id"],
                    "values": doc["embedding"],
                    "metadata": {
                        "text": doc["text"][:1000],  # Limit text size for metadata
                        **doc["metadata"]
                    }
                }
                vectors.append(vector)
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            print(f"Successfully uploaded {len(vectors)} documents to Pinecone")
            return True
            
        except Exception as e:
            print(f"Error uploading to Pinecone: {str(e)}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            search_kwargs = {
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True
            }
            
            if filter_dict:
                search_kwargs["filter"] = filter_dict
            
            results = self.index.query(**search_kwargs)
            
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    "id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "metadata": match.metadata
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching Pinecone: {str(e)}")
            return []
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by IDs"""
        try:
            self.index.delete(ids=ids)
            print(f"Successfully deleted {len(ids)} documents")
            return True
        except Exception as e:
            print(f"Error deleting documents: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            return self.index.describe_index_stats()
        except Exception as e:
            print(f"Error getting index stats: {str(e)}")
            return {}
    
    def clear_index(self) -> bool:
        """Clear all documents from index"""
        try:
            self.index.delete(delete_all=True)
            print("Successfully cleared index")
            return True
        except Exception as e:
            print(f"Error clearing index: {str(e)}")
            return False