#!/usr/bin/env python3
"""
Script to update all references from 'data/segmented' to 'data/segmented'
after renaming the directory.
"""

import os
import sys
from pathlib import Path


def update_file_references(file_path, old_ref, new_ref):
    """Update references in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if old_ref in content:
            updated_content = content.replace(old_ref, new_ref)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            return True
    except (UnicodeDecodeError, IOError) as e:
        # Skip binary files or files we can't read
        print(f"Skipping {file_path}: {e}")
    return False


def main():
    """Main function to update all references."""
    if len(sys.argv) != 1:
        print("Usage: python update_directory_references.py")
        print("This script updates references in the scripts directory only.")
        sys.exit(1)

    root_dir = Path(".")
    old_ref = "data/segmented"
    new_ref = "data/segmented"

    updated_files = []

    print(f"Updating references from '{old_ref}' to '{new_ref}' in {root_dir}")

    for root, dirs, files in os.walk(root_dir):
        # Process all files in scripts directory and subdirectories

        for file in files:
            file_path = Path(root) / file
            if update_file_references(file_path, old_ref, new_ref):
                updated_files.append(str(file_path))
                print(f"Updated: {file_path}")

    print(f"\nUpdated {len(updated_files)} files:")
    for file in updated_files:
        print(f"  {file}")


if __name__ == "__main__":
    main()
