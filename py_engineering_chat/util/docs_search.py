import re
from py_engineering_chat.util.chroma_search import search_chroma
from py_engineering_chat.util.context_model import ContextData
from py_engineering_chat.util.logger_util import get_configured_logger

def handle_docs_query(user_input: str, settings_manager=None) -> ContextData:
    logger = get_configured_logger(__name__)
    logger.debug(f"Handling docs query: {user_input}")
    docs_match = re.search(r'@docs:([\w_-]+)', user_input)
    logger.debug(f"Docs match: {docs_match}")
    if docs_match:
        collection_name = docs_match.group(1)
        context_list = search_chroma(collection_name, user_input)  # Get the list from search_chroma
        context_data = ContextData(context_description="Documetation that the user asked to be included.")
        for context in context_list:  # Iterate over the list
            context_data.add_context(context)  # Add each context to context_data
        return context_data
    return ContextData()