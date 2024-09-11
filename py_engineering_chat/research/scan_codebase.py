import os
import chromadb
from pathlib import Path
from py_engineering_chat.agents.text_summarizer import TextSummarizer
from py_engineering_chat.agents.context_evaluator import ContextEvaluator
from sentence_transformers import SentenceTransformer
import json
from dotenv import load_dotenv
from datetime import datetime
from py_engineering_chat.util.chat_settings_manager import ChatSettingsManager
from py_engineering_chat.util.logger_util import get_configured_logger  # Import the logger

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
    # Initialize logger
    logger = get_configured_logger(__name__)
    
    # Load environment variables
    load_dotenv()
    logger.debug("Environment variables loaded.")
    
    # Load settings manager
    settings_manager = ChatSettingsManager()
    settings_manager.load_settings()
    logger.debug("Settings manager loaded.")
    
    # Retrieve project directory from settings
    project_dir = settings_manager.get_setting(f'projects.{project_name}.directory')
    if not project_dir:
        logger.error(f"Directory for project '{project_name}' not found in settings.")
        raise ValueError(f"Directory for project '{project_name}' not found in settings.")
    
    # Get AI_SHADOW_DIRECTORY from environment variables
    ai_shadow_directory = os.getenv('AI_SHADOW_DIRECTORY')
    if not ai_shadow_directory:
        logger.error("AI_SHADOW_DIRECTORY environment variable is not set")
        raise ValueError("AI_SHADOW_DIRECTORY environment variable is not set")
    
    # Initialize Chroma client
    chroma_db_path = os.path.join(ai_shadow_directory, '.chroma_db')
    client = chromadb.PersistentClient(path=chroma_db_path)
    logger.debug("Chroma client initialized.")
    
    # Delete existing collection if it exists
    collection_name = f"codebase_{project_name}"
    try:
        client.delete_collection(name=collection_name)
        logger.debug(f"Existing collection '{collection_name}' deleted.")
    except ValueError:
        logger.debug(f"No existing collection '{collection_name}' to delete.")
    
    # Create new collection
    collection = client.create_collection(name=collection_name)
    logger.debug(f"New collection '{collection_name}' created.")
    
    # Initialize ContextEvaluator
    context_evaluator = ContextEvaluator()
    logger.debug("ContextEvaluator initialized.")

    # Initialize TextSummarizer only if needed
    summarizer = None if skip_summarization else TextSummarizer()
    if summarizer:
        logger.debug("TextSummarizer initialized.")
    
    # Initialize SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.debug("SentenceTransformer model initialized.")
    
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
            logger.debug(f"Skipped folder: {relative_root}")
            continue
        else:
            is_contextual, _ = context_evaluator.is_contextual(str(relative_root), "folder")
        
        if not is_contextual:
            dirs[:] = []
            folders_skipped += 1
            logger.debug(f"Non-contextual folder skipped: {relative_root}")
            continue

        # Remove ignored directories during traversal
        dirs[:] = [d for d in dirs if os.path.join(root, d) not in ALWAYS_SKIP_DIRS and d not in ALWAYS_SKIP_DIRS]
        
        for file in files:
            # Check if we've reached the max file count (if set)
            if max_files != -1 and files_processed >= max_files:
                logger.info(f"Reached maximum file count of {max_files}. Stopping scan.")
                break

            file_path = Path(root) / file
            relative_path = file_path.relative_to(project_dir)
            
            # Check if the file should be skipped
            if any(file_path.match(pattern) for pattern in ALWAYS_SKIP_FILES):
                files_skipped += 1
                logger.debug(f"Skipped file: {relative_path}")
                continue
            
            # Check if the file is likely to add context
            is_contextual, _ = context_evaluator.is_contextual(str(relative_path), "file")
            if not is_contextual:
                files_skipped += 1
                logger.debug(f"Non-contextual file skipped: {relative_path}")
                continue

            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                logger.debug(f"Reading file: {relative_path}")
                content = f.read()

            if summarizer:
                try:
                    summary = summarizer.summarize(content)
                except Exception as e:
                    logger.error(f"Error summarizing file {relative_path}: {e}")
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
            logger.debug(f"Processed file: {relative_path}")
    
    # Print summary statistics
    logger.info(f"Scan complete for project '{project_name}':")
    logger.info(f"  Files processed: {files_processed}")
    logger.info(f"  Folders skipped: {folders_skipped}")
    logger.info(f"  Files skipped: {files_skipped}")
    logger.info(f"  Total evaluations: {context_evaluator.total_evaluations}")
    logger.info(f"  Contextual ratio: {context_evaluator.contextual_ratio:.2%}")
    if skip_summarization:
        logger.info("  Summarization was skipped.")
    
    # Update max_files in the summary statistics
    if max_files == -1:
        logger.info("  Max files: No limit")
    else:
        logger.info(f"  Max files allowed: {max_files}")

    # Check if the collection is empty
    if collection.count() == 0:
        logger.warning("The collection is empty. No files were added.")

    # Load settings manager
    settings_manager = ChatSettingsManager()
    settings = settings_manager.load_settings()

    # After scanning is complete
    if collection.count() > 0:
        # Update the project's scanned status and scan date
        settings_manager.set_setting(f'projects.{project_name}.scanned', True)
        settings_manager.set_setting(f'projects.{project_name}.last_scanned', datetime.now().isoformat())
        logger.debug("Project scan status updated.")

    return collection
