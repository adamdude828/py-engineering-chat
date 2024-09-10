import re
from py_engineering_chat.util.chroma_search import search_chroma
from py_engineering_chat.util.context_model import ContextData

def handle_codebase_query(user_input: str, settings_manager) -> ContextData:
    codebase_match = re.search(r'@codebase:(.*)', user_input)
    if codebase_match:
        query = codebase_match.group(1).strip()
        current_project = settings_manager.get_setting('current_project')
        if current_project:
            collection_name = f"codebase_{current_project}"
            context = search_chroma(collection_name, query)
            context_data = ContextData()
            context_data.add_context(context)
            return context_data
        else:
            return ContextData(context=["Error: No current project set. Please set a project first."])
    return ContextData()