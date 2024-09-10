import os
import subprocess
from .chat_settings_manager import ChatSettingsManager

def add_codebase(project_name, directory, github_origin):
    """Add a codebase to the project and clone it to the shadow directory."""
    settings_manager = ChatSettingsManager()
    
    # Use set_setting instead of add_project
    settings_manager.set_setting(f'projects.{project_name}.directory', directory)
    settings_manager.set_setting(f'projects.{project_name}.github_origin', github_origin)

    # Example of using get_setting with dot notation
    project_dir = settings_manager.get_setting(f'projects.{project_name}.directory')
    print(f"Project directory: {project_dir}")

    # Clone the repository to the shadow directory
    project_shadow_dir = settings_manager.shadow_path / project_name
    
    if not project_shadow_dir.exists():
        project_shadow_dir.mkdir(parents=True, exist_ok=True)
    
    current_dir = os.getcwd()
    try:
        os.chdir(settings_manager.shadow_dir)
        subprocess.run(['git', 'clone', github_origin, project_name], check=True)
        print(f"Cloned repository for '{project_name}' to {project_shadow_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
    finally:
        os.chdir(current_dir)