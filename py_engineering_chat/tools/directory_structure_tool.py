import os
import json
from typing import Dict, Any
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
from py_engineering_chat.util.chat_settings_manager import ChatSettingsManager
from py_engineering_chat.util.logger_util import get_configured_logger

class DirectoryStructureInput(BaseModel):
    path: str = Field(description="The path to crawl for directory structure.")

class DirectoryStructureTool(BaseTool):
    name = "directory_structure"
    description = "Crawl a directory and return its structure as a JSON object"
    args_schema: type[BaseModel] = DirectoryStructureInput  # Ensure this is a class variable


    def _normalize_path(self, path: str) -> str:
        """Normalize the path based on the operating system."""
        settings_manager = ChatSettingsManager()
        is_windows = settings_manager.get_setting("is_windows", "false").lower() == "true"
        if is_windows:
            return os.path.normpath(path.replace('/', '\\'))
        return os.path.normpath(path)

    def _is_within_shadow_directory(self, path: str) -> bool:
        """Check if the path is within the shadow directory."""
        settings_manager = ChatSettingsManager()
        shadow_dir = os.path.abspath(settings_manager.get_shadow_directory())
        return os.path.commonpath([shadow_dir, path]) == shadow_dir

    def _get_directory_structure(self, path: str) -> Dict[str, Any]:
        """Recursively build the directory structure."""
        structure = {}
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    if entry.is_dir():
                        structure[entry.name] = self._get_directory_structure(entry.path)
                    elif entry.is_file():
                        structure[entry.name] = "file"
        except PermissionError:
            return "Permission denied"
        except FileNotFoundError:
            return "Directory not found"
        return structure

    def _run(self, path: str) -> str:
        """Crawl the directory and return its structure as a JSON string."""
        logger = get_configured_logger(__name__)
        logger.debug(f"Running directory structure tool with path: {path}")
        settings_manager = ChatSettingsManager()
        shadow_directory = settings_manager.get_shadow_directory()
        full_path = self._normalize_path(os.path.abspath(os.path.join(shadow_directory, path)))
        
        if not self._is_within_shadow_directory(full_path):
            return json.dumps({"error": "Access denied. Path is outside the shadow directory."})

        if not os.path.exists(full_path):
            return json.dumps({"error": f"Path '{full_path}' does not exist."})

        if not os.path.isdir(full_path):
            return json.dumps({"error": f"Path '{full_path}' is not a directory."})

        structure = self._get_directory_structure(full_path)
        return json.dumps(structure, indent=2)

    async def _arun(self, path: str) -> str:
        """Asynchronous version of the directory structure tool."""
        return self._run(path)


class FileWriteInput(BaseModel):
    path: str = Field(description="The path to the file to write to, relative to the shadow directory.")
    content: str = Field(description="The content to write to the file.")

class FileReadInput(BaseModel):
    path: str = Field(description="The path to the file to read from, relative to the shadow directory.")

class FileWriteTool(BaseTool):
    name = "file_write"
    description = "Write content to a file within the shadow directory"
    args_schema: type[BaseModel] = FileWriteInput

    def _run(self, path: str, content: str) -> str:
        """Write content to a file."""
        settings_manager = ChatSettingsManager()
        current_project = settings_manager.get_setting('current_project')
        shadow_directory = settings_manager.get_setting(f'projects.{current_project}.shadow_directory')
        full_path = os.path.abspath(os.path.join(shadow_directory, path))

        if not self._is_within_shadow_directory(full_path):
            return "Error: Access denied. Path is outside the shadow directory."

        try:
            with open(full_path, 'w') as file:
                file.write(content)
            return f"Successfully wrote to {full_path}."
        except Exception as e:
            return f"Error writing to file: {str(e)}"

    async def _arun(self, path: str, content: str) -> str:
        """Asynchronous version of the file write tool."""
        return self._run(path, content)

class FileReadTool(BaseTool):
    name = "file_read"
    description = "Read content from a file within the shadow directory"
    args_schema: type[BaseModel] = FileReadInput

    def _run(self, path: str) -> str:
        """Read content from a file."""
        settings_manager = ChatSettingsManager()
        current_project = settings_manager.get_setting('current_project')
        shadow_directory = settings_manager.get_setting(f'projects.{current_project}.shadow_directory')
        full_path = os.path.abspath(os.path.join(shadow_directory, path))

        if not self._is_within_shadow_directory(full_path):
            return "Error: Access denied. Path is outside the shadow directory."

        try:
            with open(full_path, 'r') as file:
                return file.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    async def _arun(self, path: str) -> str:
        """Asynchronous version of the file read tool."""
        return self._run(path)