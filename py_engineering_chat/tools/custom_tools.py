from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from py_engineering_chat.tools.directory_structure_tool import DirectoryStructureTool
from py_engineering_chat.tools.file_write_tool import FileWriteTool
from py_engineering_chat.tools.file_read_tool import FileReadTool
from py_engineering_chat.tools.shell_command_tool import SafeShellCommandTool
from py_engineering_chat.tools.git_create_branch import GitCreateBranchTool
from py_engineering_chat.tools.git_commit_tool import GitCommitTool

class WeatherInput(BaseModel):
    location: str = Field(description="The name of the location to get weather for. This can be a city, state, country, or any recognizable place name.")

class FakeWeatherTool(BaseTool):
    name = "fake_weather"
    description = "Get the current weather for a specified location"
    args_schema: type[BaseModel] = WeatherInput

    def _run(self, location: str) -> str:
        """Get the current weather for a specified location."""
        return f"The weather in {location} is sunny with a temperature of 72°F (22°C)."

    async def _arun(self, location: str) -> str:
        """Asynchronous version of the weather tool."""
        return self._run(location)

def get_tools():
    directory_structure_tool = DirectoryStructureTool()
    file_write_tool = FileWriteTool()
    file_read_tool = FileReadTool()
    shell_command_tool = SafeShellCommandTool()
    git_create_branch = GitCreateBranchTool()
    git_commit_tool = GitCommitTool()
    
    return [
        directory_structure_tool,
        file_write_tool,
        file_read_tool,
        shell_command_tool,
        git_create_branch,
        git_commit_tool
    ]
