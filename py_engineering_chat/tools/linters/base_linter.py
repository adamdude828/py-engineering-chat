from abc import ABC, abstractmethod

class Linter(ABC):
    @abstractmethod
    def lint_code(self, code: str) -> bool:
        pass

    @abstractmethod
    def fix_code(self, code: str, lint_output: str) -> str:
        pass

    @abstractmethod
    def lint_and_fix(self, code: str, max_attempts: int = 2) -> tuple[bool, str]:
        pass

    @abstractmethod
    def get_lint_output(self, code: str) -> str:
        pass
