import json
from pathlib import Path
import os
from dotenv import load_dotenv
import logging

class ChatSettingsManager:
    def __init__(self):
        self.shadow_dir = ChatSettingsManager.get_ai_shadow_directory()
        self.shadow_path = Path(self.shadow_dir)
        self.shadow_path.mkdir(parents=True, exist_ok=True)
        self.chat_settings_file = self.shadow_path / '.chat_settings'
        self.logger = self._initialize_logger()

    def get_project_shadow_directory(self) -> str:
        current_project = self.get_setting('current_project')
        shadow_directory = self.get_setting(f'projects.{current_project}.shadow_directory')
        return os.path.abspath(shadow_directory)

    def _initialize_logger(self) -> logging.Logger:
        """Initialize a basic logger for internal use."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.CRITICAL)  # Default to DEBUG for internal logging

        # Create a console handler for simplicity
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.CRITICAL)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    @staticmethod
    def load_env():
        """Load environment variables from a .env file."""
        root_dir = Path(__file__).resolve().parents[2]
        env_path = root_dir / '.env'
        load_dotenv(dotenv_path=env_path)

    @staticmethod
    def get_ai_shadow_directory() -> str:
        """Return the path of the AI shadow directory."""
        ChatSettingsManager.load_env()  # Ensure .env is loaded
        shadow_dir = os.getenv('AI_SHADOW_DIRECTORY', './ai_shadow')
        if not shadow_dir:
            raise ValueError("AI_SHADOW_DIRECTORY environment variable is not set")
        return shadow_dir

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
            'github_origin': github_origin,
            'scanned': False  # Indicate the codebase is not scanned
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
        self.logger.debug(f"Attempting to retrieve setting for path: {path}")
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
                self.logger.debug(f"Found key '{key}': {value}")
            else:
                self.logger.error(f"Key '{key}' not found in settings. Returning default value: {default}")
                return default
        self.logger.debug(f"Successfully retrieved value for path '{path}': {value}")
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

    def append_to_collection(self, path, value):
        """
        Append a value to a list in the settings using dot notation.
        Example: append_to_collection('projects.my_project.docs', 'langchain')
        """
        settings = self.load_settings()
        keys = path.split('.')
        current = settings
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Ensure the last key is a list
        if keys[-1] not in current:
            current[keys[-1]] = []
        elif not isinstance(current[keys[-1]], list):
            raise ValueError(f"The path '{path}' does not point to a list.")
        
        # Check for duplicates before appending
        if value not in current[keys[-1]]:
            current[keys[-1]].append(value)
            self.save_settings(settings)

    def get_shadow_directory(self) -> str:
        """Return the shadow directory path."""
        return self.shadow_directory

    def get_logger(self, name: str) -> logging.Logger:
        """Return a configured logger for the given name."""
        shadow_dir = Path(self.get_ai_shadow_directory())
        log_file_path = shadow_dir / 'chat.log'
        
        # Determine the logging level
        log_level_str = self.get_setting('log_level', 'CRITICAL').upper()
        log_level = getattr(logging, log_level_str, logging.CRITICAL)

        # Create a logger
        logger = logging.getLogger(name)
        logger.setLevel(log_level)  # Set the logging level

        # Create a file handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)

        # Create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)

        return logger

    def get_docs_options(self) -> list:
        """
        Retrieve the list of documentation options from the settings.
        """
        docs = self.get_setting('docs', [])
        return [f":{doc}" for doc in docs]
