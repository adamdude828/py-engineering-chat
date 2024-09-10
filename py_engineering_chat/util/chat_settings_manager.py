import json
from pathlib import Path
import os
from dotenv import load_dotenv

class ChatSettingsManager:
    def __init__(self):
        root_dir = Path(__file__).resolve().parents[2]
        env_path = root_dir / '.env'
        load_dotenv(dotenv_path=env_path)
        print(root_dir)

        self.shadow_dir = os.getenv('AI_SHADOW_DIRECTORY', './ai_shadow')
        if not self.shadow_dir:
            raise ValueError("AI_SHADOW_DIRECTORY environment variable is not set")
        
        self.shadow_path = Path(self.shadow_dir)
        self.shadow_path.mkdir(parents=True, exist_ok=True)
        self.chat_settings_file = self.shadow_path / '.chat_settings'

    def load_settings(self):
        if self.chat_settings_file.exists():
            with self.chat_settings_file.open('r') as f:
                return json.load(f)
        return {}

    def save_settings(self, settings):
        with self.chat_settings_file.open('w') as f:
            json.dump(settings, f, indent=2)

    def add_project(self, project_name, directory, github_origin):
        settings = self.load_settings()
        if 'projects' not in settings:
            settings['projects'] = {}
        
        settings['projects'][project_name] = {
            'directory': directory,
            'github_origin': github_origin
        }
        
        self.save_settings(settings)
        print(f"Added codebase '{project_name}' to {self.chat_settings_file}")

    def get_setting(self, path, default=None):
        """
        Get a setting value using dot notation.
        Example: get_setting('projects.my_project.directory')
        """
        settings = self.load_settings()
        keys = path.split('.')
        value = settings
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set_setting(self, path, value):
        """
        Set a setting value using dot notation.
        Example: set_setting('projects.my_project.directory', '/path/to/project')
        """
        settings = self.load_settings()
        keys = path.split('.')
        current = settings
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
        self.save_settings(settings)
        print(f"Updated setting: {path} = {value}")