# file_read_tool.py
import os
from langchain.pydantic_v1 import BaseModel, Field
from py_engineering_chat.util.logger_util import get_configured_logger
from py_engineering_chat.tools.base_tool import BaseProjectTool  # Import the base class

class FileReadInput(BaseModel):
    path: str = Field(description="The path to the file to read from, relative to the project shadow directory.")

class FileReadTool(BaseProjectTool):
    name = "file_read"
    description = "Read content from a file within the project shadow directory"
    args_schema: type[BaseModel] = FileReadInput

    def _run(self, path: str) -> str:
        """Read content from a file."""
        logger = get_configured_logger(__name__)
        
        # Remove '@' prefix if present
        path = path.lstrip('@')
        
        logger.debug(f"Reading file: {path}")
        shadow_directory = self.get_project_shadow_directory()
        full_path = os.path.abspath(os.path.join(shadow_directory, path))

        if not self._is_within_shadow_directory(full_path):
            return "Error: Access denied. Path is outside the project shadow directory."

        try:
            with open(full_path, 'r') as file:
                content = file.read()
            return content
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return f"Error reading file: {str(e)}"

    async def _arun(self, path: str) -> str:
        """Asynchronous version of the file read tool."""
        return self._run(path)

#test