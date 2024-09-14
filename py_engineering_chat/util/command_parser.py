import re  # Add this import
from py_engineering_chat.util.docs_search import handle_docs_query
from py_engineering_chat.util.codebase_search import handle_codebase_query
from py_engineering_chat.util.context_model import ContextData
from py_engineering_chat.util.logger_util import get_configured_logger
from typing import Callable, List  # Add this import
# Define a map of command patterns to handler functions
COMMAND_HANDLERS: dict[str, Callable[[str, any], ContextData]] = {
    r'@docs:(.+)': handle_docs_query,  # Updated pattern to match any character(s) after @docs:
    r'@codebase\s*(.+)': handle_codebase_query,  # Allow optional whitespace after @codebase:
}

def parse_commands(user_input: str, settings_manager) -> List[ContextData]:
    context_data_list = []
    logger = get_configured_logger(__name__)

    # Iterate over the command handlers
    for pattern, handler in COMMAND_HANDLERS.items():
        logger.debug(f"Checking pattern: {pattern} against user_input: {user_input}")
        if re.search(pattern, user_input):
            logger.debug(f"Pattern matched: {pattern}")

            context_data = handler(user_input, settings_manager)
            context_data_list.append(context_data)  # Append each matched context data

    return context_data_list  # Return the list of context data