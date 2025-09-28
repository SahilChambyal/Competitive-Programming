"""Configuration settings for the RAG agent"""

import os
from typing import Optional

class Settings:
    """Application settings"""
    
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "")
    
    # Pinecone Configuration
    pinecone_index_name: str = "org-query-agent"
    embedding_dimension: int = 1536
    
    # OpenAI Configuration
    openai_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-ada-002"
    max_tokens: int = 1000
    temperature: float = 0.7
    
    # RAG Configuration
    max_retrieved_docs: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000



settings = Settings()