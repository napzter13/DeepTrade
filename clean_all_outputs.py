#!/usr/bin/env python3
"""
cleanup_folders.py

Deletes all files in the 'debug', 'input_cache', and 'output' directories.
"""

import os
import glob

def clean_folder(folder_name):
    """
    Removes all files in the specified folder_name,
    ignoring subdirectories.
    """
    if not os.path.exists(folder_name):
        print(f"Directory '{folder_name}' does not exist. Skipping.")
        return

    files = glob.glob(os.path.join(folder_name, "*"))
    deleted_count = 0
    for fpath in files:
        if os.path.isfile(fpath):
            os.remove(fpath)
            deleted_count += 1

    print(f"Deleted {deleted_count} file(s) from '{folder_name}'.")

def main():
    folders_to_clean = ["debug", "input_cache", "output"]
    for folder in folders_to_clean:
        clean_folder(folder)

if __name__ == "__main__":
    main()
