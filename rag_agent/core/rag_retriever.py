"""RAG retrieval system"""

from typing import List, Dict, Any, Optional
from .vector_store import VectorStore
from ..utils.embedding_manager import EmbeddingManager
from ..config import settings


class RAGRetriever:
    """Handles document retrieval for RAG system"""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
    def retrieve_documents(self, query: str, top_k: int = None, filter_by_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        if top_k is None:
            top_k = settings.max_retrieved_docs
        
        # Get query embedding
        query_embedding = self.embedding_manager.get_embedding(query)
        if not query_embedding:
            return []
        
        # Prepare filter
        filter_dict = None
        if filter_by_type:
            filter_dict = {"type": {"$eq": filter_by_type}}
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        return results
    
    def retrieve_by_category(self, query: str, categories: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve documents categorized by type"""
        if categories is None:
            categories = ["pdf", "text", "chat"]
        
        categorized_results = {}
        
        for category in categories:
            results = self.retrieve_documents(
                query=query,
                top_k=3,  # Fewer per category
                filter_by_type=category
            )
            categorized_results[category] = results
        
        return categorized_results
    
    def retrieve_with_context(self, query: str, user_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Retrieve documents with user context consideration"""
        # Enhanced query with context
        enhanced_query = query
        
        if user_context:
            # Add context information to improve retrieval
            if user_context.get("role"):
                enhanced_query += f" role:{user_context['role']}"
            if user_context.get("department"):
                enhanced_query += f" department:{user_context['department']}"
        
        results = self.retrieve_documents(enhanced_query)
        
        # Add relevance scoring based on context
        if user_context:
            for result in results:
                context_score = self._calculate_context_relevance(result, user_context)
                result["context_score"] = context_score
                result["total_score"] = result["score"] * 0.7 + context_score * 0.3
        
        return results
    
    def _calculate_context_relevance(self, result: Dict[str, Any], user_context: Dict[str, Any]) -> float:
        """Calculate relevance score based on user context"""
        relevance_score = 0.0
        
        metadata = result.get("metadata", {})
        
        # Check if document source matches user's department/role
        if user_context.get("department"):
            if user_context["department"].lower() in metadata.get("source", "").lower():
                relevance_score += 0.3
        
        if user_context.get("role"):
            if user_context["role"].lower() in result.get("text", "").lower():
                relevance_score += 0.2
        
        # Boost recent chat history relevance
        if metadata.get("type") == "chat":
            relevance_score += 0.1
        
        return min(relevance_score, 1.0)
    
    def get_relevant_context(self, query: str, max_context_length: int = 2000) -> str:
        """Get relevant context text for LLM prompt"""
        results = self.retrieve_documents(query)
        
        context_parts = []
        current_length = 0
        
        for result in results:
            text = result.get("text", "")
            if current_length + len(text) <= max_context_length:
                context_parts.append(f"Source: {result.get('metadata', {}).get('source', 'Unknown')}")
                context_parts.append(text)
                context_parts.append("---")
                current_length += len(text) + 50  # Account for formatting
            else:
                break
        
        return "\n".join(context_parts)