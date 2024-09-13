# file_write_tool.py
import os
from langchain.pydantic_v1 import BaseModel, Field
from py_engineering_chat.util.logger_util import get_configured_logger
from py_engineering_chat.tools.base_tool import BaseProjectTool  # Import the base class

class FileWriteInput(BaseModel):
    path: str = Field(description="The path to the file to write to, relative to the project shadow directory.")
    content: str = Field(description="The content to write to the file.")

class FileWriteTool(BaseProjectTool):
    name = "file_write"
    description = "Write content to a file within the project shadow directory"
    args_schema: type[BaseModel] = FileWriteInput

    def _run(self, path: str, content: str) -> str:
        """Write content to a file."""
        logger = get_configured_logger(__name__)
        logger.debug(f"Writing to file: {path}")
        shadow_directory = self.get_project_shadow_directory()
        full_path = os.path.abspath(os.path.join(shadow_directory, path))

        if not self._is_within_shadow_directory(full_path):
            return "Error: Access denied. Path is outside the project shadow directory."

        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)  # Ensure the directory exists
            with open(full_path, 'w') as file:
                file.write(content)
            return f"Successfully wrote to {full_path}."
        except Exception as e:
            logger.error(f"Error writing to file: {e}")
            return f"Error writing to file: {str(e)}"

    async def _arun(self, path: str, content: str) -> str:
        """Asynchronous version of the file write tool."""
        return self._run(path, content)
