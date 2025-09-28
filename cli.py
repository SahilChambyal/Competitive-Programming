"""Command-line interface for RAG Agent"""

import asyncio
import os
import json
from pathlib import Path
from dotenv import load_dotenv

from rag_agent.core import VectorStore, QueryAgent
from rag_agent.utils import EmbeddingManager, DocumentProcessor
from rag_agent.config import settings


class RAGAgentCLI:
    """Command-line interface for the RAG Agent"""
    
    def __init__(self):
        self.embedding_manager = None
        self.vector_store = None
        self.query_agent = None
        self.document_processor = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize the agent components"""
        try:
            print("🔧 Initializing RAG Agent...")
            
            self.embedding_manager = EmbeddingManager(
                api_key=settings.openai_api_key,
                model=settings.embedding_model
            )
            
            self.vector_store = VectorStore(
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment,
                index_name=settings.pinecone_index_name
            )
            
            self.query_agent = QueryAgent(self.vector_store, self.embedding_manager)
            self.document_processor = DocumentProcessor(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
            
            self.initialized = True
            print("✅ Agent initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing agent: {str(e)}")
            return False
        
        return True
    
    def print_menu(self):
        """Print the main menu"""
        print("\n" + "="*50)
        print("🤖 RAG Query Resolution Agent")
        print("="*50)
        print("1. Ask a question")
        print("2. Upload documents")
        print("3. Upload chat history")
        print("4. Search knowledge base")
        print("5. View index statistics")
        print("6. Clear knowledge base")
        print("7. Test with authentication")
        print("0. Exit")
        print("-"*50)
    
    async def ask_question(self):
        """Handle question asking"""
        query = input("Enter your question: ").strip()
        if not query:
            print("Please enter a valid question.")
            return
        
        print("\n🤔 Processing your question...")
        
        try:
            result = self.query_agent.resolve_query(query)
            
            print(f"\n📝 Answer: {result['answer']}")
            
            if result['sources']:
                print(f"\n📚 Sources: {', '.join(result['sources'])}")
            
            if result['requires_auth']:
                print("\n🔐 This query requires authentication to access personal data.")
            
            if result['suggested_actions']:
                print(f"\n💡 Suggested actions: {', '.join(result['suggested_actions'])}")
                
        except Exception as e:
            print(f"❌ Error processing question: {str(e)}")
    
    async def upload_documents(self):
        """Handle document upload"""
        path = input("Enter path to document or directory: ").strip()
        
        if not os.path.exists(path):
            print("Path does not exist.")
            return
        
        print("\n📄 Processing documents...")
        
        try:
            if os.path.isfile(path):
                # Single file
                if path.lower().endswith('.pdf'):
                    documents = self.document_processor.process_pdf(path)
                elif path.lower().endswith(('.txt', '.md')):
                    documents = self.document_processor.process_text_file(path)
                else:
                    print("Unsupported file type. Please use PDF, TXT, or MD files.")
                    return
            else:
                # Directory
                documents = self.document_processor.process_directory(path)
            
            if documents:
                print(f"📊 Found {len(documents)} document chunks")
                
                # Embed documents
                print("🧠 Generating embeddings...")
                embedded_docs = self.embedding_manager.embed_documents(documents)
                
                if embedded_docs:
                    # Store in vector database
                    print("💾 Storing in knowledge base...")
                    success = self.vector_store.upsert_documents(embedded_docs)
                    
                    if success:
                        print(f"✅ Successfully processed {len(documents)} document chunks!")
                    else:
                        print("❌ Failed to store documents.")
                else:
                    print("❌ Failed to generate embeddings.")
            else:
                print("No documents found or processed.")
                
        except Exception as e:
            print(f"❌ Error uploading documents: {str(e)}")
    
    async def upload_chat_history(self):
        """Handle chat history upload"""
        path = input("Enter path to JSON file with chat history: ").strip()
        
        if not os.path.exists(path):
            print("File does not exist.")
            return
        
        try:
            with open(path, 'r') as f:
                chat_data = json.load(f)
            
            if not isinstance(chat_data, list):
                print("Chat history should be a list of chat records.")
                return
            
            print(f"\n💬 Processing {len(chat_data)} chat records...")
            
            documents = self.document_processor.process_chat_history(chat_data)
            
            if documents:
                embedded_docs = self.embedding_manager.embed_documents(documents)
                
                if embedded_docs:
                    success = self.vector_store.upsert_documents(embedded_docs)
                    
                    if success:
                        print(f"✅ Successfully processed {len(documents)} chat records!")
                    else:
                        print("❌ Failed to store chat history.")
                else:
                    print("❌ Failed to generate embeddings.")
            else:
                print("No chat records processed.")
                
        except Exception as e:
            print(f"❌ Error uploading chat history: {str(e)}")
    
    async def search_knowledge_base(self):
        """Handle knowledge base search"""
        query = input("Enter search query: ").strip()
        if not query:
            return
        
        try:
            results = self.query_agent.rag_retriever.retrieve_documents(query, top_k=3)
            
            if results:
                print(f"\n🔍 Found {len(results)} relevant documents:")
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. Score: {result['score']:.3f}")
                    print(f"   Source: {result['metadata'].get('source', 'Unknown')}")
                    print(f"   Text: {result['text'][:200]}...")
            else:
                print("No relevant documents found.")
                
        except Exception as e:
            print(f"❌ Error searching: {str(e)}")
    
    async def show_index_stats(self):
        """Show index statistics"""
        try:
            stats = self.vector_store.get_index_stats()
            print("\n📊 Index Statistics:")
            print(json.dumps(stats, indent=2))
        except Exception as e:
            print(f"❌ Error getting stats: {str(e)}")
    
    async def clear_knowledge_base(self):
        """Clear the knowledge base"""
        confirm = input("Are you sure you want to clear the entire knowledge base? (yes/no): ").strip().lower()
        
        if confirm == 'yes':
            try:
                success = self.vector_store.clear_index()
                if success:
                    print("✅ Knowledge base cleared successfully!")
                else:
                    print("❌ Failed to clear knowledge base.")
            except Exception as e:
                print(f"❌ Error clearing knowledge base: {str(e)}")
        else:
            print("Operation cancelled.")
    
    async def test_with_auth(self):
        """Test queries with authentication context"""
        print("\n🔐 Testing with authentication context")
        
        user_context = {
            "user_id": "test_user_123",
            "token": "mock_token",
            "organization_api": "https://api.example-org.com",
            "role": "student",
            "department": "Computer Science"
        }
        
        query = input("Enter a query that might need personal data: ").strip()
        if not query:
            return
        
        try:
            result = self.query_agent.resolve_query(query, user_context)
            
            print(f"\n📝 Answer: {result['answer']}")
            
            if result['requires_auth']:
                print("\n🔐 This query requires authentication.")
                print("Note: In a real scenario, the agent would make API calls to fetch personal data.")
            
            if result['agentic_data']:
                print("\n🤖 Agentic data would be fetched:")
                for key, data in result['agentic_data'].items():
                    print(f"   {key}: {data}")
                    
        except Exception as e:
            print(f"❌ Error processing authenticated query: {str(e)}")
    
    async def run(self):
        """Main CLI loop"""
        # Load environment variables
        load_dotenv()
        
        # Check required environment variables
        required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            print("Please set these in your .env file")
            return
        
        # Initialize agent
        if not await self.initialize():
            return
        
        # Main loop
        while True:
            self.print_menu()
            
            try:
                choice = input("Enter your choice (0-7): ").strip()
                
                if choice == '0':
                    print("👋 Goodbye!")
                    break
                elif choice == '1':
                    await self.ask_question()
                elif choice == '2':
                    await self.upload_documents()
                elif choice == '3':
                    await self.upload_chat_history()
                elif choice == '4':
                    await self.search_knowledge_base()
                elif choice == '5':
                    await self.show_index_stats()
                elif choice == '6':
                    await self.clear_knowledge_base()
                elif choice == '7':
                    await self.test_with_auth()
                else:
                    print("Invalid choice. Please try again.")
                    
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {str(e)}")
                input("Press Enter to continue...")


async def main():
    """Main function"""
    cli = RAGAgentCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())