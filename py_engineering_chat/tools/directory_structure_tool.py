import os
import json
from typing import Dict, Any
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

class DirectoryStructureInput(BaseModel):
    path: str = Field(description="The path to crawl for directory structure.")

class DirectoryStructureTool(BaseTool):
    name = "directory_structure"
    description = "Crawl a directory and return its structure as a JSON object"
    args_schema: type[BaseModel] = DirectoryStructureInput
    shadow_directory: str = Field(default_factory=lambda: os.getenv("AI_SHADOW_DIRECTORY", "/path/to/default/shadow/directory"))
    is_windows: bool = Field(default_factory=lambda: os.getenv("IS_WINDOWS", "false").lower() == "true")
    ignored_directories: list = Field(default_factory=lambda: os.getenv("IGNORED_DIRECTORIES", "").split(","))

    def _normalize_path(self, path: str) -> str:
        """Normalize the path based on the operating system."""
        if self.is_windows:
            return os.path.normpath(path.replace('/', '\\'))
        return os.path.normpath(path)

    def _get_directory_structure(self, path: str) -> Dict[str, Any]:
        """Recursively build the directory structure."""
        structure = {}
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    if entry.is_dir() and entry.name not in self.ignored_directories:
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
        full_path = self._normalize_path(os.path.join(self.shadow_directory, path))
        
        if not os.path.exists(full_path):
            return json.dumps({"error": f"Path '{full_path}' does not exist."})

        if not os.path.isdir(full_path):
            return json.dumps({"error": f"Path '{full_path}' is not a directory."})

        structure = self._get_directory_structure(full_path)
        return json.dumps(structure, indent=2)

    async def _arun(self, path: str) -> str:
        """Asynchronous version of the directory structure tool."""
        return self._run(path)