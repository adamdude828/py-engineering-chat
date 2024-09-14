from pydantic import BaseModel
from typing import List

class ContextData(BaseModel):
    context: List[str] = []
    context_description: str = ""

    def __init__(self, context=None, context_description=""):
        super().__init__(context=context if context else [], context_description=context_description)

    def add_context(self, new_context: List[str], limit: int = 3):
        self.context.extend(new_context[:limit])

    def toString(self) -> str:
        return f"Description: {self.context_description}\nContext: {'; '.join(self.context)}"