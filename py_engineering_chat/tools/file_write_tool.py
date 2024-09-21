# file_write_tool.py
import os
from langchain.pydantic_v1 import BaseModel, Field
from py_engineering_chat.util.logger_util import get_configured_logger
from py_engineering_chat.tools.base_tool import BaseProjectTool

class FileWriteInput(BaseModel):
    path: str = Field(description="The path to the file to modify, relative to the project shadow directory.")
    content: str = Field(description="The new content to write to the file.")

class FileWriteTool(BaseProjectTool):
    name = "file_write"
    description = "Write content to a file within the project shadow directory"
    args_schema: type[BaseModel] = FileWriteInput

    def _run(self, path: str, content: str) -> str:
        """Write new content to a file with confirmation."""
        logger = get_configured_logger(__name__)
        logger.debug(f"Modifying file: {path}")
        shadow_directory = self.get_project_shadow_directory()
        full_path = os.path.abspath(os.path.join(shadow_directory, path))

        if not self._is_within_shadow_directory(full_path):
            return "Error: Access denied. Path is outside the project shadow directory."

        try:
            # Read the current content of the file
            with open(full_path, 'r') as file:
                current_content = file.read()

            # Truncate content for display
            display_current_content = (current_content[:200] + '...') if len(current_content) > 200 else current_content
            display_new_content = (content[:200] + '...') if len(content) > 200 else content

            # Confirm with the user
            print(f"Current content of {path}:\n{display_current_content}\n")
            print(f"New content to be written:\n{display_new_content}\n")
            confirmation = input(f"Do you want to replace the entire content of {path} with the new content? (yes/no): ")
            if confirmation.lower() != 'yes':
                return "Modification cancelled by user."

            # Write the new content to the file
            with open(full_path, 'w') as file:
                file.write(content)

            return f"Successfully modified {full_path}."
        except Exception as e:
            logger.error(f"Error modifying file: {e}")
            return f"Error modifying file: {str(e)}"

    async def _arun(self, path: str, content: str) -> str:
        """Asynchronous version of the file write tool."""
        return self._run(path, content)
