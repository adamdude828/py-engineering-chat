# file_write_tool.py
import os
from langchain.pydantic_v1 import BaseModel, Field
from py_engineering_chat.util.logger_util import get_configured_logger
from py_engineering_chat.tools.base_tool import BaseProjectTool  # Import the base class

class FileWriteInput(BaseModel):
    path: str = Field(description="The path to the file to modify, relative to the project shadow directory.")
    search_text: str = Field(description="The text to search for in the file.")
    replace_text: str = Field(description="The text to replace the search text with.")

class FileWriteTool(BaseProjectTool):
    name = "file_write"
    description = "Write content to a file within the project shadow directory"
    args_schema: type[BaseModel] = FileWriteInput

    def _run(self, path: str, search_text: str, replace_text: str) -> str:
        """Search and replace text in a file with confirmation."""
        logger = get_configured_logger(__name__)
        logger.debug(f"Modifying file: {path}")
        shadow_directory = self.get_project_shadow_directory()
        full_path = os.path.abspath(os.path.join(shadow_directory, path))

        if not self._is_within_shadow_directory(full_path):
            return "Error: Access denied. Path is outside the project shadow directory."

        try:
            with open(full_path, 'r') as file:
                content = file.read()

            new_content = content.replace(search_text, replace_text)

            # Truncate search and replace text for display
            display_search_text = (search_text[:200] + '...') if len(search_text) > 200 else search_text
            display_replace_text = (replace_text[:200] + '...') if len(replace_text) > 200 else replace_text

            # Confirm with the user
            confirmation = input(f"Do you want to replace '{display_search_text}' with '{display_replace_text}' in {path}? (yes/no): ")
            if confirmation.lower() != 'yes':
                return "Modification cancelled by user."

            with open(full_path, 'w') as file:
                file.write(new_content)

            return f"Successfully modified {full_path}."
        except Exception as e:
            logger.error(f"Error modifying file: {e}")
            return f"Error modifying file: {str(e)}"

    async def _arun(self, path: str, search_text: str, replace_text: str) -> str:
        """Asynchronous version of the search and replace tool."""
        return self._run(path, search_text, replace_text)
