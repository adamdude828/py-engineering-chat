import os
import subprocess
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from py_engineering_chat.tools.base_tool import BaseProjectTool

class ShellCommandInput(BaseModel):
    command: str = Field(description="The shell command to execute.")

class SafeShellCommandTool(BaseProjectTool):
    name = "safe_shell_command"
    description = "Execute shell commands within a specified shadow directory, useful for running git commands and understanding pending changes. Returns True if the command executed successfully but the user didn't want to add the output to the chat. Returns False if the user didn't confirm the command execution."
    args_schema: type[BaseModel] = ShellCommandInput
    shadow_directory: str = Field(default_factory=lambda: os.getenv("SHADOW_DIRECTORY", "/path/to/default/shadow/directory"))

    def _run(self, command: str) -> str | bool:
        """Execute a shell command within the shadow directory."""
        self.shadow_directory = self.get_project_shadow_directory()

        if not os.path.exists(self.shadow_directory):
            return f"Error: Shadow directory '{self.shadow_directory}' does not exist."

        # Confirm before executing the command
        confirm = input(f"\033[92mAre you sure you want to execute the command: '{command}'? (yes/y): \033[0m")
        if confirm.lower() not in ['yes', 'y']:
            return False

        try:
            # Change to the shadow directory
            os.chdir(self.shadow_directory)
            
            # Execute the command
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            # Print stdout and stderr
            print("\033[94mCommand output:\033[0m")
            print(result.stdout)
            print("\033[91mCommand errors:\033[0m")
            print(result.stderr)
            
            # Ask if the user wants to add the output to the chat
            add_to_chat = input("\033[92mDo you want to add this output to the chat? (yes/y): \033[0m")
            
            if add_to_chat.lower() in ['yes', 'y']:
                return f"Command output:\n{result.stdout}\nCommand errors:\n{result.stderr}"
            else:
                return True  # Command executed successfully, but output not added to chat
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(f"\033[91m{error_message}\033[0m")
            return error_message

    async def _arun(self, command: str) -> str | bool:
        """Asynchronous version of the shell command tool."""
        return self._run(command)