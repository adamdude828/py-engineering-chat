import re
from py_engineering_chat.util.chroma_search import search_chroma
from py_engineering_chat.util.context_model import ContextData

def handle_docs_query(user_input: str, settings_manager=None) -> ContextData:
    docs_match = re.search(r'@docs:(\w+)', user_input)
    if docs_match:
        collection_name = docs_match.group(1)
        context = search_chroma(collection_name, user_input)
        context_data = ContextData()
        context_data.add_context(context)
        return context_data
    return ContextData()