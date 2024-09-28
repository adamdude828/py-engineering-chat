import pylint.lint
from io import StringIO
import sys
from py_engineering_chat.util.logger_util import get_configured_logger
from py_engineering_chat.util.tiered_memory import TieredMemory
from langchain_community.chat_models import ChatOpenAI
from .base_linter import Linter

class PythonLinter(Linter):
    def __init__(self):
        self.logger = get_configured_logger(__name__)
        self.tiered_memory = TieredMemory()
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4")

    def lint_code(self, code: str) -> bool:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            pylint.lint.Run(['-', '--output-format=text'], do_exit=False, stdin=code)
            lint_output = sys.stdout.getvalue()
            return "Your code has been rated at 10.00/10" in lint_output
        finally:
            sys.stdout = old_stdout

    def fix_code(self, code: str, lint_output: str) -> str:
        context = self.tiered_memory.get_context(code, n_results=3)
        context_str = "\n".join([f"{item['metadata']['role']}: {item['content']}" for item in context])
        
        prompt = f"""
        You are a Python expert. The following code failed linting:

        {code}

        Linting output:
        {lint_output}

        Recent context:
        {context_str}

        Please fix the code to pass linting. Only return the fixed code, no explanations.
        """

        response = self.llm.invoke(prompt)
        return response.content

    def lint_and_fix(self, code: str, max_attempts: int = 2) -> tuple[bool, str]:
        for attempt in range(max_attempts):
            if self.lint_code(code):
                return True, code

            self.logger.info(f"Linting failed. Attempt {attempt + 1} to fix.")
            lint_output = self.get_lint_output(code)
            code = self.fix_code(code, lint_output)

        return self.lint_code(code), code

    def get_lint_output(self, code: str) -> str:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            pylint.lint.Run(['-', '--output-format=text'], do_exit=False, stdin=code)
            return sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
