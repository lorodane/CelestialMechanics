import os
import json
import subprocess
import numpy as np

def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "Unknown"

metadata = {
    "commit": get_git_revision_hash(),
    "params": {
        "eccentricity": 0.5,
        "tolerance": 1e-12,
        "t_max": 800
    },
    "description": "Template experiment run script"
}

# Ensure data and plots directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Save metadata
with open('data/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"Running experiment with code version: {metadata['commit']}")
# Add your core logic here
# results = my_integrator_call(...)
# np.save('data/results.npy', results)
