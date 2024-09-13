import os
import pygit2
from langchain.pydantic_v1 import BaseModel, Field
from py_engineering_chat.util.logger_util import get_configured_logger
from py_engineering_chat.tools.base_tool import BaseProjectTool

class GitCommitInput(BaseModel):
    message: str = Field(description="The commit message.")
    author_name: str = Field(default="Author", description="The name of the author.")
    author_email: str = Field(default="author@example.com", description="The email of the author.")

class GitCommitTool(BaseProjectTool):
    name = "git_commit"
    description = "Commit changes to the git repository in the project shadow directory"
    args_schema: type[BaseModel] = GitCommitInput

    def _run(self, message: str, author_name: str = "Author", author_email: str = "author@example.com") -> str:
        """Commit changes to the git repository."""
        logger = get_configured_logger(__name__)
        shadow_directory = self.get_project_shadow_directory()
        repo_path = os.path.abspath(shadow_directory)

        try:
            repo = pygit2.Repository(repo_path)
            index = repo.index
            index.add_all()  # Stage all changes
            index.write()
            tree = index.write_tree()

            author = pygit2.Signature(author_name, author_email)
            committer = pygit2.Signature(author_name, author_email)
            if repo.head_is_unborn:
                parents = []
            else:
                parents = [repo.head.target]
            oid = repo.create_commit('HEAD', author, committer, message, tree, parents)
            return f"Committed changes with OID: {oid}"
        except Exception as e:
            logger.error(f"Error committing changes: {e}")
            return f"Error committing changes: {str(e)}"

    async def _arun(self, message: str, author_name: str = "Author", author_email: str = "author@example.com") -> str:
        """Asynchronous version of the git commit tool."""
        return self._run(message, author_name, author_email)
