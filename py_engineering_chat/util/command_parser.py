import re  # Add this import
from typing import Dict, Any, Callable
from py_engineering_chat.util.docs_search import handle_docs_query
from py_engineering_chat.util.codebase_search import handle_codebase_query
from py_engineering_chat.util.context_model import ContextData

# Define a map of command patterns to handler functions
COMMAND_HANDLERS: Dict[str, Callable[[str, Any], ContextData]] = {
    r'@docs:(\w+)': handle_docs_query,
    r'@codebase:(.*)': handle_codebase_query,
}

def parse_commands(user_input: str, settings_manager) -> Dict[str, Any]:
    combined_context = ContextData()

    # Iterate over the command handlers
    for pattern, handler in COMMAND_HANDLERS.items():
        if re.search(pattern, user_input):
            context_data = handler(user_input, settings_manager)
            combined_context.add_context(context_data.context)

    return "\n".join(combined_context.context) if combined_context.context else "No additional context provided."