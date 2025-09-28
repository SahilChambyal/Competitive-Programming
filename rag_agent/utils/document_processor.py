"""Document processing utilities for RAG system"""

import os
from typing import List, Dict, Any
from pathlib import Path
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class DocumentProcessor:
    """Handles document processing and chunking for RAG system"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF file and return chunked documents"""
        documents = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text += page.extract_text() + "\n"
                
                # Create chunks
                chunks = self.text_splitter.split_text(text)
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": file_path,
                            "page": page_num + 1,
                            "chunk": i + 1,
                            "type": "pdf"
                        }
                    )
                    documents.append(doc)
                    
        except Exception as e:
            print(f"Error processing PDF {file_path}: {str(e)}")
            
        return documents
    
    def process_text_file(self, file_path: str) -> List[Document]:
        """Process text file and return chunked documents"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                
            chunks = self.text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path,
                        "chunk": i + 1,
                        "type": "text"
                    }
                )
                documents.append(doc)
                
        except Exception as e:
            print(f"Error processing text file {file_path}: {str(e)}")
            
        return documents
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """Process all supported files in directory"""
        documents = []
        directory = Path(directory_path)
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                if file_path.suffix.lower() == '.pdf':
                    documents.extend(self.process_pdf(str(file_path)))
                elif file_path.suffix.lower() in ['.txt', '.md']:
                    documents.extend(self.process_text_file(str(file_path)))
        
        return documents
    
    def process_chat_history(self, chat_data: List[Dict[str, Any]]) -> List[Document]:
        """Process chat history data"""
        documents = []
        
        for i, chat in enumerate(chat_data):
            content = f"Query: {chat.get('query', '')}\nResponse: {chat.get('response', '')}"
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": "chat_history",
                    "chat_id": chat.get("id", i),
                    "timestamp": chat.get("timestamp", ""),
                    "type": "chat"
                }
            )
            documents.append(doc)
            
        return documents