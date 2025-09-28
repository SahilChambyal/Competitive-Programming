"""Embedding management utilities"""

from typing import List, Dict, Any
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import numpy as np


class EmbeddingManager:
    """Manages text embeddings using OpenAI"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model=model
        )
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            return []
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            print(f"Error getting embeddings: {str(e)}")
            return []
    
    def embed_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Embed documents and return with metadata"""
        embedded_docs = []
        
        texts = [doc.page_content for doc in documents]
        embeddings = self.get_embeddings(texts)
        
        for doc, embedding in zip(documents, embeddings):
            if embedding:  # Only add if embedding was successful
                embedded_docs.append({
                    "id": f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk', 0)}",
                    "text": doc.page_content,
                    "embedding": embedding,
                    "metadata": doc.metadata
                })
        
        return embedded_docs
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0