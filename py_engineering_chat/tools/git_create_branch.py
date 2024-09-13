import os
import pygit2
from langchain.pydantic_v1 import BaseModel, Field
from py_engineering_chat.util.logger_util import get_configured_logger
from py_engineering_chat.tools.base_tool import BaseProjectTool

class GitCreateBranchInput(BaseModel):
    branch_name: str = Field(description="The name of the new branch.")
    start_point: str = Field(default='HEAD', description="The starting point for the new branch.")

class GitCreateBranchTool(BaseProjectTool):
    name = "git_create_branch"
    description = "Create a new branch in the git repository in the project shadow directory"
    args_schema: type[BaseModel] = GitCreateBranchInput

    def _run(self, branch_name: str, start_point: str = 'HEAD') -> str:
        """Create a new branch."""
        logger = get_configured_logger(__name__)
        shadow_directory = self.get_project_shadow_directory()
        repo_path = os.path.abspath(shadow_directory)

        try:
            repo = pygit2.Repository(repo_path)
            if start_point == 'HEAD':
                target = repo.head.target
            else:
                try:
                    target = repo.revparse_single(start_point).oid
                except KeyError:
                    return f"Error: Invalid start point '{start_point}'."

            # Check if branch already exists
            if branch_name in repo.branches.local:
                return f"Error: Branch '{branch_name}' already exists."

            # Create the branch
            repo.branches.local.create(branch_name, repo.get(target))
            return f"Branch '{branch_name}' created successfully."
        except Exception as e:
            logger.error(f"Error creating branch: {e}")
            return f"Error creating branch: {str(e)}"

    async def _arun(self, branch_name: str, start_point: str = 'HEAD') -> str:
        """Asynchronous version of the git create branch tool."""
        return self._run(branch_name, start_point)
