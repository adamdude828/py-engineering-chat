import os
import pygit2
from langchain.pydantic_v1 import BaseModel, Field
from py_engineering_chat.util.logger_util import get_configured_logger
from py_engineering_chat.tools.base_tool import BaseProjectTool

class GitMergeInput(BaseModel):
    from_branch: str = Field(description="The name of the branch to merge from.")

class GitMergeTool(BaseProjectTool):
    name = "git_merge"
    description = "Merge a branch into the current branch in the git repository in the project shadow directory"
    args_schema: type[BaseModel] = GitMergeInput

    def _run(self, from_branch: str) -> str:
        """Merge a branch into the current branch."""
        logger = get_configured_logger(__name__)
        shadow_directory = self.get_project_shadow_directory()
        repo_path = os.path.abspath(shadow_directory)

        try:
            repo = pygit2.Repository(repo_path)
            current_branch = repo.head.shorthand
            logger.debug(f"Current branch: {current_branch}")
            logger.debug(f"Merging from branch: {from_branch}")
            # Find the branch to merge from
            merge_branch = repo.branches.get(from_branch)
            if merge_branch is None:
                return f"Error: Branch '{from_branch}' does not exist."
            merge_commit = repo.get(merge_branch.target)

            # Perform the merge
            repo.merge(merge_commit.oid)

            if repo.index.conflicts is not None:
                # Handle conflicts
                repo.state_cleanup()
                return "Error: Merge conflicts detected. Manual resolution required."
            else:
                # Create a merge commit
                author = pygit2.Signature("Author", "author@example.com")
                committer = pygit2.Signature("Author", "author@example.com")
                tree = repo.index.write_tree()
                parents = [repo.head.target, merge_commit.oid]
                message = f"Merge branch '{from_branch}' into '{current_branch}'"
                oid = repo.create_commit('HEAD', author, committer, message, tree, parents)
                repo.state_cleanup()
                return f"Merge completed with OID: {oid}"
        except Exception as e:
            logger.error(f"Error merging branches: {e}")
            return f"Error merging branches: {str(e)}"

    async def _arun(self, from_branch: str) -> str:
        """Asynchronous version of the git merge tool."""
        return self._run(from_branch)
