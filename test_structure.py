"""Simple structure test without external dependencies"""

import os
import sys

def test_project_structure():
    """Test that all required files and directories exist"""
    base_path = "/home/runner/work/Competitive-Programming/Competitive-Programming"
    
    required_files = [
        "rag_agent/__init__.py",
        "rag_agent/config/__init__.py",
        "rag_agent/config/settings.py",
        "rag_agent/core/__init__.py",
        "rag_agent/core/vector_store.py",
        "rag_agent/core/rag_retriever.py",
        "rag_agent/core/agentic_handler.py",
        "rag_agent/core/query_agent.py",
        "rag_agent/utils/__init__.py",
        "rag_agent/utils/document_processor.py",
        "rag_agent/utils/embedding_manager.py",
        "rag_agent/api/__init__.py",
        "rag_agent/api/main.py",
        "requirements.txt",
        "README.md",
        "main.py",
        "cli.py",
        ".env.example",
        ".gitignore"
    ]
    
    required_dirs = [
        "rag_agent",
        "rag_agent/config",
        "rag_agent/core", 
        "rag_agent/utils",
        "rag_agent/api",
        "rag_agent/tests",
        "examples",
        "examples/sample_documents",
        "examples/chat_history"
    ]
    
    missing_files = []
    missing_dirs = []
    
    # Check directories
    for dir_path in required_dirs:
        full_path = os.path.join(base_path, dir_path)
        if not os.path.exists(full_path):
            missing_dirs.append(dir_path)
    
    # Check files
    for file_path in required_files:
        full_path = os.path.join(base_path, file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
    
    # Report results
    print("🧪 Project Structure Test")
    print("=" * 50)
    
    print(f"✅ Directories: {len(required_dirs) - len(missing_dirs)}/{len(required_dirs)} present")
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
    
    print(f"✅ Files: {len(required_files) - len(missing_files)}/{len(required_files)} present")
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
    
    # Test imports without external dependencies
    print("\n🔧 Testing Core Module Imports")
    try:
        sys.path.append(base_path)
        
        # Test configuration module
        from rag_agent.config.settings import Settings
        print("✅ Settings class can be imported")
        
        # Test basic structure
        print("✅ Basic project structure is valid")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
    
    success = len(missing_files) == 0 and len(missing_dirs) == 0
    
    if success:
        print("\n🎉 All structure tests passed!")
    else:
        print("\n❌ Some structure tests failed")
    
    return success

if __name__ == "__main__":
    test_project_structure()