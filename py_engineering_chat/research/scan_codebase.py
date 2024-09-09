import os
import chromadb
from pathlib import Path
from py_engineering_chat.agents.text_summarizer import TextSummarizer
from py_engineering_chat.agents.context_evaluator import ContextEvaluator
from sentence_transformers import SentenceTransformer
import json
from dotenv import load_dotenv

# Configuration for directories to always skip
ALWAYS_SKIP_DIRS = {'.git', 'node_modules', 'vendor', 'build', 'dist', 'venv', '__pycache__'}

# Configuration for files to always skip (including wildcards)
ALWAYS_SKIP_FILES = {
    # Version control
    '.gitignore', '.gitattributes', '.hgignore', '.svnignore',
    # Environment and configuration
    '.env', '.env.*', '*.cfg', '*.conf', '*.ini', '*.config',
    # Package managers
    'package-lock.json', 'yarn.lock', 'Pipfile.lock', 'poetry.lock',
    # Docker
    '.dockerignore', 'Dockerfile', 'docker-compose.yml',
    # IDE and editor files
    '.vscode', '.idea', '*.sublime-project', '*.sublime-workspace',
    # Compiled files and build artifacts
    '*.pyc', '*.pyo', '*.pyd', '*.class', '*.dll', '*.exe', '*.o', '*.so',
    '*.egg-info', '*.egg', '*.whl',
    # Logs and databases
    '*.log', '*.sql', '*.sqlite',
    # OS generated files
    '.DS_Store', 'Thumbs.db',
    # Documentation
    '*.md', '*.rst', 'LICENSE', 'README*',
    # Other common ignore patterns
    '*.bak', '*.swp', '*.swo', '*.tmp', '*.temp',
    # Add more patterns as needed
}

def scan_codebase(project_name, skip_summarization=False, max_files=-1):
    # Load environment variables
    load_dotenv()
    
    # Load .chat_settings
    with open('.chat_settings', 'r') as f:
        settings = json.load(f)
    
    if project_name not in settings['projects']:
        raise ValueError(f"Project '{project_name}' not found in .chat_settings")
    
    project_dir = settings['projects'][project_name]['directory']
    
    # Get ignored directories from .env
    ignored_dirs = os.getenv('IGNORED_DIRECTORIES', '').split(',')
    ignored_dirs = [dir.strip() for dir in ignored_dirs if dir.strip()]
    
    # Initialize Chroma client
    client = chromadb.PersistentClient(path="./.chroma_db")
    
    # Delete existing collection if it exists
    collection_name = f"codebase_{project_name}"
    try:
        client.delete_collection(name=collection_name)
    except ValueError:
        pass
    
    # Create new collection
    collection = client.create_collection(name=collection_name)
    
    # Initialize ContextEvaluator
    context_evaluator = ContextEvaluator()

    # Initialize TextSummarizer only if needed
    summarizer = None if skip_summarization else TextSummarizer()
    
    # Initialize SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    files_processed = 0
    folders_skipped = 0
    files_skipped = 0

    # Walk through the project directory
    for root, dirs, files in os.walk(project_dir):
        relative_root = Path(root).relative_to(project_dir)
        if relative_root == Path('.'):
            is_contextual = True
        elif any(part in ALWAYS_SKIP_DIRS for part in relative_root.parts):
            dirs[:] = []
            folders_skipped += 1
            continue
        else:
            is_contextual, _ = context_evaluator.is_contextual(str(relative_root), "folder")
        
        if not is_contextual:
            dirs[:] = []
            folders_skipped += 1
            continue

        # Remove ignored directories during traversal
        dirs[:] = [d for d in dirs if os.path.join(root, d) not in ignored_dirs and d not in ALWAYS_SKIP_DIRS]
        
        for file in files:
            # Check if we've reached the max file count (if set)
            if max_files != -1 and files_processed >= max_files:
                print(f"Reached maximum file count of {max_files}. Stopping scan.")
                break

            file_path = Path(root) / file
            relative_path = file_path.relative_to(project_dir)
            
            # Check if the file should be skipped
            if any(file_path.match(pattern) for pattern in ALWAYS_SKIP_FILES):
                files_skipped += 1
                continue
            
            # Check if the file is likely to add context
            is_contextual, _ = context_evaluator.is_contextual(str(relative_path), "file")
            if not is_contextual:
                files_skipped += 1
                continue

            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if (summarizer):
                try:
                    summary = summarizer.summarize(content)
                except Exception as e:
                    print(f"Error calling summarizer: {e}")
                    summary = content[:1000]  # Fallback to using first 1000 characters
            else:
                summary = content[:1000]  # Use first 1000 characters as summary

            
            # Calculate embedding
            embedding = model.encode(summary).tolist()
            
            # Add to collection
            collection.add(
                ids=[str(relative_path)],
                documents=[content],
                embeddings=[embedding],
                metadatas=[{"path": str(relative_path), "summary": summary}]
            )
            files_processed += 1
    
    # Print summary statistics
    print(f"Scan complete for project '{project_name}':")
    print(f"  Files processed: {files_processed}")
    print(f"  Folders skipped: {folders_skipped}")
    print(f"  Files skipped: {files_skipped}")
    print(f"  Total evaluations: {context_evaluator.total_evaluations}")
    print(f"  Contextual ratio: {context_evaluator.contextual_ratio:.2%}")
    if skip_summarization:
        print("  Summarization was skipped.")
    
    # Update max_files in the summary statistics
    if max_files == -1:
        print("  Max files: No limit")
    else:
        print(f"  Max files allowed: {max_files}")

    # Check if the collection is empty
    if collection.count() == 0:
        print("Warning: The collection is empty. No files were added.")

    return collection
