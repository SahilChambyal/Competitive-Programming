"""FastAPI application for RAG Agent"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import os
import tempfile

from ..core import VectorStore, QueryAgent
from ..utils import EmbeddingManager, DocumentProcessor
from ..config import settings

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    user_context: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[str]
    requires_auth: bool
    suggested_actions: List[str]
    agentic_data: Optional[Dict[str, Any]] = None

class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    documents_processed: int

class UserContext(BaseModel):
    user_id: Optional[str] = None
    token: Optional[str] = None
    organization_api: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None

# Initialize components
app = FastAPI(
    title="RAG-Powered Query Resolution Agent",
    description="AI Agent for organizational query resolution with RAG and agentic capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (initialized on startup)
vector_store = None
embedding_manager = None
query_agent = None
document_processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global vector_store, embedding_manager, query_agent, document_processor
    
    try:
        # Initialize components
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
        document_processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        print("RAG Agent initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing RAG Agent: {str(e)}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG-Powered Query Resolution Agent",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if components are initialized
        if not all([vector_store, embedding_manager, query_agent]):
            raise HTTPException(status_code=503, detail="Service not ready")
        
        # Check vector store connectivity
        stats = vector_store.get_index_stats()
        
        return {
            "status": "healthy",
            "components": {
                "vector_store": "connected",
                "embedding_manager": "ready",
                "query_agent": "ready"
            },
            "index_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def resolve_query(request: QueryRequest):
    """Main query resolution endpoint"""
    try:
        if not query_agent:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        # Resolve query
        result = query_agent.resolve_query(
            query=request.query,
            user_context=request.user_context
        )
        
        # Add to knowledge base if successful
        if not result.get("requires_auth") or result.get("agentic_data"):
            query_agent.add_chat_to_knowledge_base(
                query=request.query,
                response=result["answer"],
                user_id=request.user_context.get("user_id") if request.user_context else None
            )
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query resolution failed: {str(e)}")

@app.post("/upload-documents", response_model=DocumentUploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents for the knowledge base"""
    try:
        if not all([document_processor, embedding_manager, vector_store]):
            raise HTTPException(status_code=503, detail="Service not ready")
        
        total_documents = 0
        
        for file in files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Process document based on file type
                if file.filename.lower().endswith('.pdf'):
                    documents = document_processor.process_pdf(temp_file_path)
                elif file.filename.lower().endswith(('.txt', '.md')):
                    documents = document_processor.process_text_file(temp_file_path)
                else:
                    continue  # Skip unsupported files
                
                if documents:
                    # Embed documents
                    embedded_docs = embedding_manager.embed_documents(documents)
                    
                    # Store in vector database
                    if embedded_docs:
                        vector_store.upsert_documents(embedded_docs)
                        total_documents += len(documents)
                
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
        
        return DocumentUploadResponse(
            success=True,
            message=f"Successfully processed {total_documents} document chunks",
            documents_processed=total_documents
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")

@app.post("/upload-chat-history")
async def upload_chat_history(chat_data: List[Dict[str, Any]]):
    """Upload chat history to knowledge base"""
    try:
        if not all([document_processor, embedding_manager, vector_store]):
            raise HTTPException(status_code=503, detail="Service not ready")
        
        # Process chat history
        documents = document_processor.process_chat_history(chat_data)
        
        if documents:
            # Embed documents
            embedded_docs = embedding_manager.embed_documents(documents)
            
            # Store in vector database
            if embedded_docs:
                vector_store.upsert_documents(embedded_docs)
                return {
                    "success": True,
                    "message": f"Successfully processed {len(documents)} chat records",
                    "records_processed": len(documents)
                }
        
        return {
            "success": False,
            "message": "No valid chat records found"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat history upload failed: {str(e)}")

@app.get("/search")
async def search_documents(query: str, top_k: int = 5):
    """Search documents in the knowledge base"""
    try:
        if not query_agent:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        results = query_agent.rag_retriever.retrieve_documents(query, top_k)
        
        return {
            "query": query,
            "results": results,
            "total": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.delete("/clear-knowledge-base")
async def clear_knowledge_base():
    """Clear all documents from the knowledge base"""
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        success = vector_store.clear_index()
        
        if success:
            return {"success": True, "message": "Knowledge base cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear knowledge base")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear operation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )