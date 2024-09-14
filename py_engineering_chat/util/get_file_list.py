import os
from py_engineering_chat.util.chat_settings_manager import ChatSettingsManager

def get_file_list():
    directory = ChatSettingsManager().get_project_shadow_directory()
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Remove print statements used for debugging
            file_path = os.path.join(root, file)
            file_list.append(os.path.relpath(file_path, directory))  # Ensure paths are relative to the base directory
    return file_list
