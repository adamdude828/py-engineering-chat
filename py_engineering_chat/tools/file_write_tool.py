import os
from langchain.pydantic_v1 import BaseModel, Field
from py_engineering_chat.util.logger_util import get_configured_logger
from py_engineering_chat.tools.base_tool import BaseProjectTool
from py_engineering_chat.tools.linting_router import LintingRouter

class FileWriteInput(BaseModel):
    path: str = Field(description="The path to the file to modify, relative to the project shadow directory.")
    content: str = Field(description="The new content to write to the file.")

class FileWriteTool(BaseProjectTool):
    name = "file_write"
    description = "Write content to a file within the project shadow directory"
    args_schema: type[BaseModel] = FileWriteInput

    def _run(self, path: str, content: str) -> str:
        logger = get_configured_logger(__name__)
        logger.debug(f"Attempting to modify file: {path}")
        shadow_directory = self.get_project_shadow_directory()
        full_path = os.path.abspath(os.path.join(shadow_directory, path))

        if not self._is_within_shadow_directory(full_path):
            return "Error: Access denied. Path is outside the project shadow directory."

        try:
            # Simple confirmation with just the filename
            confirmation = input(f"Do you want to modify the file '{path}'? (yes/no): ")
            if confirmation.lower() != 'yes':
                return "Modification cancelled by user."

            # Construct LintingRouter here
            linting_router = LintingRouter()

            # Lint and fix the code using the router
            linting_passed, fixed_content = linting_router.lint_and_fix(path, content)

            if not linting_passed:
                logger.warning("Linting failed after fix attempts. Write operation cancelled.")
                return False

            # Write the fixed content to the file
            with open(full_path, 'w') as file:
                file.write(fixed_content)

            return f"Successfully modified and linted {full_path}."
        except Exception as e:
            logger.error(f"Error modifying file: {e}")
            return False

    async def _arun(self, path: str, content: str) -> str:
        """Asynchronous version of the file write tool."""
        return self._run(path, content)
