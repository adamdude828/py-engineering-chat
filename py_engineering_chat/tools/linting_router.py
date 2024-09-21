import os
from py_engineering_chat.tools.linters.python_linter import PythonLinter
from py_engineering_chat.util.logger_util import get_configured_logger

class LintingRouter:
    def __init__(self):
        self.logger = get_configured_logger(__name__)
        self.linters = {
            '.py': PythonLinter()
        }

    def get_linter(self, file_path: str):
        _, file_extension = os.path.splitext(file_path)
        linter = self.linters.get(file_extension)
        if linter is None:
            self.logger.warning(f"No linter found for file extension: {file_extension}")
        return linter

    def lint_and_fix(self, file_path: str, content: str) -> tuple[bool, str]:
        linter = self.get_linter(file_path)
        if linter is None:
            return True, content  # If no linter is found, assume the content is fine
        return linter.lint_and_fix(content)
