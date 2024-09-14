from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from py_engineering_chat.util.get_file_list import get_file_list
from py_engineering_chat.util.chat_settings_manager import ChatSettingsManager

file_list = get_file_list()
chat_settings_manager = ChatSettingsManager()

class FileCompleter(Completer):
    def get_completions(self, document, complete_event):
        text_before_cursor = document.text_before_cursor
        at_index = text_before_cursor.rfind('@')
        if at_index == -1:
            return

        # Extract the prefix after the last '@'
        prefix = text_before_cursor[at_index + 1:]

        # Check if the prefix is 'docs'
        if prefix.lower().startswith('docs:'):
            matches = chat_settings_manager.get_docs_options()
            # Calculate the start position to replace only after 'docs:'
            start_position = at_index + len('docs:') - len(text_before_cursor)
        else:
            # Perform a case-insensitive substring match
            matches = [f for f in file_list if prefix.lower() in f.lower()]

            # If no prefix after '@', list all files and directories
            if not prefix:
                matches = file_list

            # Calculate the start position to replace from '@' to the cursor
            start_position = at_index - len(text_before_cursor) + 1

        for match in matches:
            yield Completion(match, start_position=start_position)
