# base_tool.py
import os
from langchain.tools import BaseTool
from py_engineering_chat.util.chat_settings_manager import ChatSettingsManager

class BaseProjectTool(BaseTool):
    def get_project_shadow_directory(self) -> str:
        settings_manager = ChatSettingsManager()
        return settings_manager.get_project_shadow_directory()

    def _is_within_shadow_directory(self, path: str) -> bool:
        shadow_dir = self.get_project_shadow_directory()
        return os.path.commonpath([shadow_dir, path]) == shadow_dir
