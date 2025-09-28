# RAG-Powered Query Resolution Agent

An intelligent AI agent designed for private organizations and government offices to handle queries efficiently using RAG (Retrieval-Augmented Generation) capabilities powered by Pinecone and agentic capabilities for API interactions.

## 🚀 Features

### Core Capabilities
- **RAG-powered Query Resolution**: Uses official documents, PDFs, policy documents, and chat history to answer queries
- **Agentic API Interactions**: Fetches personal data from organization databases using authentication tokens
- **Multi-format Document Processing**: Supports PDF, TXT, and Markdown files
- **Vector Database Storage**: Uses Pinecone for efficient similarity search
- **Chat History Learning**: Learns from previous human responses to improve future answers

### Agentic Features
- **User Authentication**: Handles session tokens and API authentication
- **Personal Data Retrieval**: Fetches individual profiles, academic records, attendance data
- **Request Submission**: Can submit requests and track their status
- **Context-Aware Responses**: Considers user role and department for personalized answers

## 🏗️ Architecture

```
RAG Agent
├── Core Components
│   ├── QueryAgent (Main orchestrator)
│   ├── VectorStore (Pinecone integration)
│   ├── RAGRetriever (Document retrieval)
│   └── AgenticHandler (API interactions)
├── Utilities
│   ├── DocumentProcessor (PDF/text processing)
│   └── EmbeddingManager (OpenAI embeddings)
├── API Layer
│   └── FastAPI endpoints
└── Configuration
    └── Environment-based settings
```

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone account and API key
- Organization API endpoints (for agentic features)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Competitive-Programming
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Configure your .env file**:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=your_pinecone_environment_here
   ```

## 🚀 Usage

### 1. Command Line Interface

Run the interactive CLI:
```bash
python cli.py
```

Features available in CLI:
- Ask questions
- Upload documents
- Upload chat history
- Search knowledge base
- View statistics
- Test authentication features

### 2. API Server

Start the FastAPI server:
```bash
python -m uvicorn rag_agent.api.main:app --reload
```

The API will be available at `http://localhost:8000`

#### API Endpoints

- `POST /query` - Submit a query for resolution
- `POST /upload-documents` - Upload documents to knowledge base
- `POST /upload-chat-history` - Upload chat history
- `GET /search` - Search the knowledge base
- `GET /health` - Health check
- `DELETE /clear-knowledge-base` - Clear all data

### 3. Programmatic Usage

```python
from rag_agent.core import VectorStore, QueryAgent
from rag_agent.utils import EmbeddingManager
from rag_agent.config import settings

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

# Simple query
result = query_agent.resolve_query("What are the office hours?")
print(result["answer"])

# Query with user context (for personal data)
user_context = {
    "user_id": "12345",
    "token": "user_session_token",
    "organization_api": "https://api.organization.com",
    "role": "student",
    "department": "Computer Science"
}

result = query_agent.resolve_query("What are my grades?", user_context)
print(result["answer"])
```

## 📊 Document Processing

### Supported Formats
- **PDF files**: Automatically extracted and chunked
- **Text files**: .txt and .md files
- **Chat history**: JSON format with query-response pairs

### Example Chat History Format
```json
[
  {
    "id": "chat_001",
    "query": "How do I apply for leave?",
    "response": "To apply for leave, you need to fill out Form-123...",
    "timestamp": "2024-01-15T10:30:00Z",
    "user_id": "user123"
  }
]
```

## 🔐 Authentication & Agentic Features

The agent can interact with organization APIs to fetch personal data:

### User Context Structure
```python
user_context = {
    "user_id": "student_12345",
    "token": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "organization_api": "https://university-api.edu",
    "role": "student",
    "department": "Computer Science"
}
```

### Supported Agentic Actions
- **fetch_user_profile**: Get basic user information
- **fetch_academic_records**: Get grades, GPA, courses
- **fetch_attendance**: Get attendance records
- **submit_request**: Submit requests to the organization

## 🔧 Configuration

Key configuration options in `rag_agent/config/settings.py`:

```python
# Pinecone settings
pinecone_index_name: str = "org-query-agent"
embedding_dimension: int = 1536

# OpenAI settings
openai_model: str = "gpt-3.5-turbo"
embedding_model: str = "text-embedding-ada-002"
max_tokens: int = 1000
temperature: float = 0.7

# RAG settings
max_retrieved_docs: int = 5
chunk_size: int = 1000
chunk_overlap: int = 200
```

## 📈 Use Cases

### Government Offices
- **Citizen Services**: Answer queries about procedures, requirements, forms
- **Policy Information**: Provide information from policy documents
- **Status Tracking**: Check application/request status

### Educational Institutions
- **Student Support**: Answer academic policy questions
- **Personal Records**: Fetch grades, attendance, transcripts
- **Administrative Help**: Guide through enrollment, fee payment processes

### Corporate Organizations
- **Employee Support**: HR policies, benefits information
- **IT Helpdesk**: Technical documentation and troubleshooting
- **Administrative Tasks**: Leave applications, expense reporting

## 🧪 Testing

Run the main application to test basic functionality:
```bash
python main.py
```

This will initialize all components and run sample queries to verify everything is working.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the documentation above
- Look at example usage in `main.py` and `cli.py`
- Create an issue in the repository

## 🔮 Future Enhancements

- Multi-language support
- Voice interface integration
- Advanced analytics and reporting
- Integration with more vector databases
- Improved security features
- Custom authentication providers