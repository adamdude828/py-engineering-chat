# directory_structure_tool.py
import os
import json
from typing import Dict, Any
from langchain.pydantic_v1 import BaseModel, Field
from py_engineering_chat.util.logger_util import get_configured_logger
from py_engineering_chat.tools.base_tool import BaseProjectTool  # Import the base class
from py_engineering_chat.util.chat_settings_manager import ChatSettingsManager

class DirectoryStructureInput(BaseModel):
    path: str = Field(description="The path to crawl for directory structure.")

class DirectoryStructureTool(BaseProjectTool):
    name = "directory_structure"
    description = """
    Crawl a directory and return its structure as a JSON object. This is helpful if you are trying to find the location of a file.
    """
    args_schema: type[BaseModel] = DirectoryStructureInput  # Ensure this is a class variable

    def _normalize_path(self, path: str) -> str:
        """Normalize the path based on the operating system."""
        settings_manager = ChatSettingsManager()
        is_windows = settings_manager.get_setting("is_windows", "false").lower() == "true"
        if is_windows:
            return os.path.normpath(path.replace('/', '\\'))
        return os.path.normpath(path)

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
        shadow_directory = self.get_project_shadow_directory()
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
