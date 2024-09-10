from pydantic import BaseModel
from typing import List

class ContextData(BaseModel):
    context: List[str] = []

    def add_context(self, new_context: List[str], limit: int = 3):
        self.context.extend(new_context[:limit])