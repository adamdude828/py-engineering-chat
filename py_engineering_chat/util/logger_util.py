import logging
from pathlib import Path
from .chat_settings_manager import ChatSettingsManager

def get_configured_logger(name: str) -> logging.Logger:
    """Return a configured logger that logs to a file in the shadow directory."""
    settings_manager = ChatSettingsManager()
    shadow_dir = Path(ChatSettingsManager.get_ai_shadow_directory())
    log_file_path = shadow_dir / 'chat.log'
    
    # Determine the logging level
    log_level_str = settings_manager.get_setting('log_level', 'CRITICAL').upper()
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