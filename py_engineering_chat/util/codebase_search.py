import re
from py_engineering_chat.util.chroma_search import search_chroma
from py_engineering_chat.util.context_model import ContextData
from py_engineering_chat.util.logger_util import get_configured_logger

def handle_codebase_query(user_input: str, settings_manager) -> ContextData:
    logger = get_configured_logger(__name__)
    logger.debug(f"Handling codebase query: {user_input}")
    
    query = user_input.strip()
    current_project = settings_manager.get_setting('current_project')
    if current_project:
        collection_name = f"codebase_{current_project}"
        context_list = search_chroma(collection_name, query)  # Ensure this returns a list
        context_data = ContextData(context_description="Result of a codebase search based on users query.")
        for context in context_list:  # Iterate over the list
            context_data.add_context(context)  # Add each context to context_data
        return context_data
    else:
        logger.error("No current project set. Please set a project first.")
        return ContextData(context=["Error: No current project set. Please set a project first."])