import os
import glob

def replace_in_file(filepath, old, new):
    with open(filepath, 'r') as f:
        content = f.read()
    
    if old in content:
        print(f"Updating {filepath}")
        content = content.replace(old, new)
        with open(filepath, 'w') as f:
            f.write(content)

base_dirs = [
    "scripts/segmentation",
    "scripts/analysis",
    "scripts/visualization",
    "scripts/utils"
]

for d in base_dirs:
    for filepath in glob.glob(os.path.join(d, "*.py")):
        replace_in_file(filepath, "../data", "../../data")
