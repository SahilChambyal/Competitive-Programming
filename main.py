"""Main entry point for RAG Agent application"""

import asyncio
import os
from dotenv import load_dotenv

from rag_agent.core import VectorStore, QueryAgent
from rag_agent.utils import EmbeddingManager, DocumentProcessor
from rag_agent.config import settings


async def main():
    """Main application function"""
    # Load environment variables
    load_dotenv()
    
    print("🚀 Starting RAG-powered Query Resolution Agent...")
    
    # Check required environment variables
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file or environment")
        return
    
    try:
        # Initialize components
        print("🔧 Initializing components...")
        
        embedding_manager = EmbeddingManager(
            api_key=settings.openai_api_key,
            model=settings.embedding_model
        )
        
        vector_store = VectorStore(
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_environment,
            index_name=settings.pinecone_index_name
        )
        
        query_agent = QueryAgent(vector_store, embedding_manager)
        
        print("✅ Components initialized successfully!")
        
        # Test the agent with sample queries
        print("\n🧪 Testing agent with sample queries...")
        
        test_queries = [
            "What are the office hours?",
            "How do I apply for leave?",
            "What documents do I need for admission?"
        ]
        
        for query in test_queries:
            print(f"\n❓ Query: {query}")
            result = query_agent.resolve_query(query)
            print(f"📝 Answer: {result['answer'][:200]}...")
            if result['sources']:
                print(f"📚 Sources: {', '.join(result['sources'])}")
        
        print("\n🎉 Agent is ready! You can now:")
        print("1. Start the API server: python -m uvicorn rag_agent.api.main:app --reload")
        print("2. Use the CLI interface: python cli.py")
        print("3. Upload documents via the API or programmatically")
        
    except Exception as e:
        print(f"❌ Error initializing agent: {str(e)}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    asyncio.run(main())