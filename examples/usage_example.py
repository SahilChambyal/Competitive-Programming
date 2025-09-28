"""Example usage of the RAG Agent"""

import asyncio
import os
from dotenv import load_dotenv

from rag_agent.core import VectorStore, QueryAgent
from rag_agent.utils import EmbeddingManager, DocumentProcessor
from rag_agent.config import settings


async def main():
    """Demonstrate RAG Agent capabilities"""
    
    # Load environment variables
    load_dotenv()
    
    print("🚀 RAG Agent Example Usage")
    print("=" * 50)
    
    # Initialize components
    print("\n1. Initializing components...")
    try:
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
        document_processor = DocumentProcessor()
        
        print("✅ Components initialized successfully!")
        
        # Upload sample documents
        print("\n2. Processing sample documents...")
        
        # Process policy document
        policy_path = "examples/sample_documents/policy.md"
        if os.path.exists(policy_path):
            documents = document_processor.process_text_file(policy_path)
            if documents:
                embedded_docs = embedding_manager.embed_documents(documents)
                vector_store.upsert_documents(embedded_docs)
                print(f"✅ Processed policy document: {len(documents)} chunks")
        
        # Process chat history
        chat_path = "examples/chat_history/sample_chats.json"
        if os.path.exists(chat_path):
            import json
            with open(chat_path, 'r') as f:
                chat_data = json.load(f)
            
            chat_documents = document_processor.process_chat_history(chat_data)
            if chat_documents:
                embedded_chat_docs = embedding_manager.embed_documents(chat_documents)
                vector_store.upsert_documents(embedded_chat_docs)
                print(f"✅ Processed chat history: {len(chat_documents)} records")
        
        # Test queries
        print("\n3. Testing query resolution...")
        
        test_queries = [
            "What are the office hours?",
            "How do I apply for leave?",
            "What is the minimum attendance required?",
            "How can I reset my password?",
            "What are my grades?"  # This will require authentication
        ]
        
        for query in test_queries:
            print(f"\n❓ Query: {query}")
            
            # Basic query
            result = query_agent.resolve_query(query)
            print(f"📝 Answer: {result['answer'][:200]}...")
            
            if result['sources']:
                print(f"📚 Sources: {', '.join(result['sources'])}")
            
            if result['requires_auth']:
                print("🔐 This query requires authentication")
                
                # Test with authentication context
                user_context = {
                    "user_id": "student_12345",
                    "token": "mock_bearer_token",
                    "organization_api": "https://api.university.edu",
                    "role": "student",
                    "department": "Computer Science"
                }
                
                print("   Testing with authentication context...")
                auth_result = query_agent.resolve_query(query, user_context)
                print(f"   🔐 Authenticated answer: {auth_result['answer'][:150]}...")
            
            print("-" * 50)
        
        # Test search functionality
        print("\n4. Testing search functionality...")
        
        search_results = query_agent.rag_retriever.retrieve_documents("office hours", top_k=3)
        print(f"🔍 Found {len(search_results)} relevant documents for 'office hours':")
        
        for i, result in enumerate(search_results, 1):
            print(f"   {i}. Score: {result['score']:.3f}")
            print(f"      Source: {result['metadata'].get('source', 'Unknown')}")
            print(f"      Text: {result['text'][:100]}...")
        
        # Show index statistics
        print("\n5. Index statistics...")
        stats = vector_store.get_index_stats()
        print(f"📊 Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"📊 Index fullness: {stats.get('index_fullness', 0)}")
        
        print("\n🎉 Example completed successfully!")
        print("\nNext steps:")
        print("- Run 'python cli.py' for interactive CLI")
        print("- Run 'python -m uvicorn rag_agent.api.main:app --reload' for API server")
        print("- Upload your own documents and test with real queries")
        
    except Exception as e:
        print(f"❌ Error during example execution: {str(e)}")
        print("Please check your configuration and API keys")


if __name__ == "__main__":
    asyncio.run(main())