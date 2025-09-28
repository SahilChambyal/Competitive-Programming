"""Main Query Resolution Agent"""

import openai
from typing import Dict, Any, List, Optional
from .vector_store import VectorStore
from .rag_retriever import RAGRetriever
from .agentic_handler import AgenticHandler
from ..utils.embedding_manager import EmbeddingManager
from ..config import settings


class QueryAgent:
    """Main AI Agent for query resolution"""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.rag_retriever = RAGRetriever(vector_store, embedding_manager)
        self.agentic_handler = AgenticHandler()
        
        # Initialize OpenAI
        openai.api_key = settings.openai_api_key
        
        self.system_prompt = """You are an intelligent query resolution agent for organizations and government offices. 
        You help users resolve queries using official documents, policies, and previous interactions.
        
        Your capabilities include:
        1. Answering questions using official documentation and policies
        2. Fetching personal information for authenticated users
        3. Helping with requests and applications
        4. Providing guidance based on previous similar queries
        
        Always be helpful, accurate, and professional. If you need to fetch personal data, 
        explain what you're doing and ensure the user has proper authentication."""
    
    def resolve_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Main method to resolve user queries"""
        
        # Determine query type and required actions
        query_analysis = self._analyze_query(query, user_context)
        
        response = {
            "query": query,
            "requires_auth": query_analysis.get("requires_auth", False),
            "suggested_actions": query_analysis.get("actions", [])
        }
        
        # Get relevant context from RAG
        rag_context = self._get_rag_context(query, user_context)
        
        # Handle agentic tasks if needed
        agentic_data = None
        if query_analysis.get("requires_auth") and user_context and user_context.get("token"):
            agentic_data = self._handle_agentic_tasks(query_analysis, user_context)
        
        # Generate response using LLM
        llm_response = self._generate_llm_response(
            query=query,
            rag_context=rag_context,
            agentic_data=agentic_data,
            user_context=user_context
        )
        
        response.update({
            "answer": llm_response,
            "sources": self._extract_sources(rag_context),
            "agentic_data": agentic_data
        })
        
        return response
    
    def _analyze_query(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze query to determine required actions"""
        query_lower = query.lower()
        
        analysis = {
            "requires_auth": False,
            "actions": [],
            "category": "general"
        }
        
        # Check for personal data requests
        personal_keywords = ["my", "profile", "grades", "marks", "attendance", "performance", "records"]
        if any(keyword in query_lower for keyword in personal_keywords):
            analysis["requires_auth"] = True
            analysis["category"] = "personal"
            
            if "profile" in query_lower:
                analysis["actions"].append("fetch_user_profile")
            if any(word in query_lower for word in ["grades", "marks", "academic", "gpa"]):
                analysis["actions"].append("fetch_academic_records")
            if "attendance" in query_lower:
                analysis["actions"].append("fetch_attendance")
        
        # Check for request submission
        request_keywords = ["submit", "apply", "request", "application"]
        if any(keyword in query_lower for keyword in request_keywords):
            analysis["actions"].append("submit_request")
            analysis["category"] = "request"
        
        return analysis
    
    def _get_rag_context(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant context using RAG"""
        if user_context:
            return self.rag_retriever.retrieve_with_context(query, user_context)
        else:
            return self.rag_retriever.retrieve_documents(query)
    
    def _handle_agentic_tasks(self, query_analysis: Dict[str, Any], user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agentic tasks like API calls"""
        agentic_results = {}
        
        token = user_context.get("token")
        organization_api = user_context.get("organization_api")
        user_id = user_context.get("user_id")
        
        if not all([token, organization_api, user_id]):
            return {"error": "Missing authentication information"}
        
        # Execute required actions
        for action in query_analysis.get("actions", []):
            if action == "fetch_user_profile":
                result = self.agentic_handler.fetch_user_profile(token, organization_api, user_id)
                agentic_results["profile"] = result
            
            elif action == "fetch_academic_records":
                result = self.agentic_handler.fetch_academic_records(token, organization_api, user_id)
                agentic_results["academics"] = result
            
            elif action == "fetch_attendance":
                result = self.agentic_handler.fetch_attendance(token, organization_api, user_id)
                agentic_results["attendance"] = result
        
        return agentic_results
    
    def _generate_llm_response(self, query: str, rag_context: List[Dict[str, Any]], 
                              agentic_data: Optional[Dict[str, Any]] = None,
                              user_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using LLM"""
        
        # Prepare context
        context_text = self.rag_retriever.get_relevant_context(query)
        
        # Add agentic data to context
        agentic_context = ""
        if agentic_data:
            for key, data in agentic_data.items():
                if data.get("success"):
                    agentic_context += f"\n{key.upper()} DATA: {data.get('summary', str(data.get('data', '')))}"
        
        # Prepare messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
Context from documents:
{context_text}

{agentic_context}

User query: {query}

Please provide a helpful and accurate response based on the available information.
"""}
        ]
        
        try:
            response = openai.ChatCompletion.create(
                model=settings.openai_model,
                messages=messages,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"
    
    def _extract_sources(self, rag_context: List[Dict[str, Any]]) -> List[str]:
        """Extract source information from RAG context"""
        sources = []
        for item in rag_context:
            source = item.get("metadata", {}).get("source")
            if source and source not in sources:
                sources.append(source)
        return sources
    
    def add_chat_to_knowledge_base(self, query: str, response: str, user_id: str = None) -> bool:
        """Add successful query-response pair to knowledge base"""
        try:
            chat_data = [{
                "id": f"chat_{user_id}_{hash(query + response)}",
                "query": query,
                "response": response,
                "user_id": user_id,
                "timestamp": "now"  # Should be actual timestamp
            }]
            
            # Process and embed
            from ..utils.document_processor import DocumentProcessor
            processor = DocumentProcessor()
            documents = processor.process_chat_history(chat_data)
            
            embedded_docs = self.embedding_manager.embed_documents(documents)
            
            # Store in vector database
            return self.vector_store.upsert_documents(embedded_docs)
            
        except Exception as e:
            print(f"Error adding chat to knowledge base: {str(e)}")
            return False