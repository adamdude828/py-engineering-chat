import os
import subprocess
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from py_engineering_chat.tools.base_tool import BaseProjectTool

class ShellCommandInput(BaseModel):
    command: str = Field(description="The shell command to execute.")

class SafeShellCommandTool(BaseProjectTool):  # Inherit from BaseProjectTool
    name = "safe_shell_command"
    description = "Execute shell commands within a specified shadow directory, useful for running git commands and understanding pending changes"
    args_schema: type[BaseModel] = ShellCommandInput
    shadow_directory: str = Field(default_factory=lambda: os.getenv("SHADOW_DIRECTORY", "/path/to/default/shadow/directory"))

    def _run(self, command: str) -> str:
        """Execute a shell command within the shadow directory."""
        self.shadow_directory = self.get_project_shadow_directory()  # Use method from BaseProjectTool

        if not os.path.exists(self.shadow_directory):
            return f"Error: Shadow directory '{self.shadow_directory}' does not exist."

        # Confirm before executing the command
        confirm = input(f"Are you sure you want to execute the command: '{command}'? (yes/no): ")
        if confirm.lower() != 'yes':
            return "Command execution cancelled by user."

        try:
            # Change to the shadow directory
            os.chdir(self.shadow_directory)
            
            # Execute the command
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Error executing command: {e.stderr}"
        except Exception as e:
            return f"An error occurred: {str(e)}"

    async def _arun(self, command: str) -> str:
        """Asynchronous version of the shell command tool."""
        return self._run(command)