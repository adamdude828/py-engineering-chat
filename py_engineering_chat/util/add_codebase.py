import json
from pathlib import Path
import os
import subprocess

def add_codebase(project_name, directory, github_origin):
    """Add a codebase to the project and clone it to the shadow directory."""
    chat_settings_file = Path('.chat_settings')
    
    if chat_settings_file.exists():
        with chat_settings_file.open('r') as f:
            settings = json.load(f)
    else:
        settings = {}
    
    if 'projects' not in settings:
        settings['projects'] = {}
    
    settings['projects'][project_name] = {
        'directory': directory,
        'github_origin': github_origin
    }
    
    with chat_settings_file.open('w') as f:
        json.dump(settings, f, indent=2)
    
    print(f"Added codebase '{project_name}' to .chat_settings")

    # Clone the repository to the shadow directory
    shadow_dir = os.getenv('AI_SHADOW_DIRECTORY', './ai_shadow')
    project_shadow_dir = os.path.join(shadow_dir, project_name)
    
    if not os.path.exists(project_shadow_dir):
        os.makedirs(project_shadow_dir)
    
    current_dir = os.getcwd()
    try:
        os.chdir(shadow_dir)
        subprocess.run(['git', 'clone', github_origin, project_name], check=True)
        print(f"Cloned repository for '{project_name}' to {project_shadow_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
    finally:
        os.chdir(current_dir)