import os
import sys
import json
import subprocess
import datetime

# Ensure the working directory is the folder where this script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "Unknown"

def is_repo_dirty():
    try:
        status = subprocess.check_output(['git', 'status', '--porcelain']).decode('ascii').strip()
        return status != ""
    except Exception:
        return "Unknown"

# --- CONFIGURATION (Single Source of Truth) ---
# TODO: Add experiment description to README.md
# TODO: Update these parameters for your new experiment!
# TODO: Record any experimental conclusions in README.md
params = {
    "eccentricity": 0.5,
    "tolerance": 1e-12,
    "t_max": 800
}
# ----------------------------------------------

metadata = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "commit": get_git_revision_hash(),
    "repo_is_dirty": is_repo_dirty(),
    "params": params
}

# Ensure data and plots directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Save metadata in the experiment root (outside data folder)
with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"Running experiment | Date: {metadata['timestamp']} | Version: {metadata['commit']}")
if metadata["repo_is_dirty"]:
    print("WARNING: Uncommitted changes detected in the repository.")

# ----------------------------------------------
# EXPERIMENT CODE GOES HERE


