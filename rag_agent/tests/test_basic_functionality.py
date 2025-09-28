"""Basic functionality tests for RAG Agent components"""

import unittest
from unittest.mock import Mock, patch
import os
import tempfile

from rag_agent.utils.document_processor import DocumentProcessor
from rag_agent.utils.embedding_manager import EmbeddingManager
from rag_agent.core.agentic_handler import AgenticHandler


class TestDocumentProcessor(unittest.TestCase):
    """Test document processing functionality"""
    
    def setUp(self):
        self.processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    
    def test_process_text_file(self):
        """Test text file processing"""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document. It contains multiple sentences. "
                   "This should be split into chunks based on the configuration.")
            temp_path = f.name
        
        try:
            documents = self.processor.process_text_file(temp_path)
            
            self.assertGreater(len(documents), 0)
            for doc in documents:
                self.assertIn('source', doc.metadata)
                self.assertIn('chunk', doc.metadata)
                self.assertIn('type', doc.metadata)
                self.assertEqual(doc.metadata['type'], 'text')
                
        finally:
            os.unlink(temp_path)
    
    def test_process_chat_history(self):
        """Test chat history processing"""
        chat_data = [
            {
                "id": "test_1",
                "query": "What are office hours?",
                "response": "Office hours are 9 AM to 5 PM",
                "timestamp": "2024-01-01T10:00:00Z"
            }
        ]
        
        documents = self.processor.process_chat_history(chat_data)
        
        self.assertEqual(len(documents), 1)
        self.assertIn("Query:", documents[0].page_content)
        self.assertIn("Response:", documents[0].page_content)
        self.assertEqual(documents[0].metadata['type'], 'chat')


class TestAgenticHandler(unittest.TestCase):
    """Test agentic handler functionality"""
    
    def setUp(self):
        self.handler = AgenticHandler()
    
    def test_supported_actions(self):
        """Test that supported actions are defined"""
        expected_actions = [
            "fetch_user_profile",
            "fetch_academic_records", 
            "fetch_attendance",
            "fetch_performance_data",
            "submit_request",
            "check_request_status"
        ]
        
        for action in expected_actions:
            self.assertIn(action, self.handler.supported_actions)
    
    def test_generate_profile_summary(self):
        """Test profile summary generation"""
        profile_data = {
            "name": "John Doe",
            "role": "student",
            "department": "Computer Science",
            "email": "john.doe@university.edu"
        }
        
        summary = self.handler._generate_profile_summary(profile_data)
        
        self.assertIn("John Doe", summary)
        self.assertIn("student", summary)
        self.assertIn("Computer Science", summary)
    
    def test_generate_academic_summary(self):
        """Test academic summary generation"""
        academic_data = {
            "gpa": "3.8",
            "year": "Senior",
            "major": "Computer Science",
            "credits_completed": "120"
        }
        
        summary = self.handler._generate_academic_summary(academic_data)
        
        self.assertIn("3.8", summary)
        self.assertIn("Senior", summary)
        self.assertIn("Computer Science", summary)
    
    @patch('requests.Session.post')
    def test_submit_request(self, mock_post):
        """Test request submission"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": "req_123",
            "status": "submitted"
        }
        mock_post.return_value = mock_response
        
        request_data = {
            "type": "transcript",
            "description": "Official transcript request",
            "priority": "high"
        }
        
        result = self.handler.submit_request(
            token="test_token",
            organization_api="https://api.test.com",
            user_id="user123",
            request_data=request_data
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["request_id"], "req_123")


class TestBasicIntegration(unittest.TestCase):
    """Test basic integration between components"""
    
    def test_document_to_embedding_flow(self):
        """Test the flow from document processing to embedding preparation"""
        # Create processor
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        
        # Create sample chat data
        chat_data = [
            {
                "id": "test_chat",
                "query": "How do I reset password?",
                "response": "Contact IT support at ext 1234",
                "timestamp": "2024-01-01T10:00:00Z"
            }
        ]
        
        # Process documents
        documents = processor.process_chat_history(chat_data)
        
        # Verify structure for embedding
        self.assertGreater(len(documents), 0)
        
        for doc in documents:
            self.assertTrue(hasattr(doc, 'page_content'))
            self.assertTrue(hasattr(doc, 'metadata'))
            self.assertIsInstance(doc.page_content, str)
            self.assertIsInstance(doc.metadata, dict)


if __name__ == '__main__':
    unittest.main()