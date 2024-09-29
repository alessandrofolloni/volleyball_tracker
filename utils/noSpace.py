import os
import argparse


def remove_spaces_in_filenames(directory, recursive=False):
    """
    Removes spaces from filenames in the specified directory.

    Args:
        directory (str): The directory containing files to rename.
        recursive (bool): If True, process subdirectories recursively.
    """
    if recursive:
        # Walk through all subdirectories
        for root, dirs, files in os.walk(directory):
            for filename in files:
                old_path = os.path.join(root, filename)
                new_filename = filename.replace(' ', '_')
                new_path = os.path.join(root, new_filename)
                if old_path != new_path:
                    os.rename(old_path, new_path)
                    print(f"Renamed: '{old_path}' -> '{new_path}'")
    else:
        # Only process files in the specified directory
        for filename in os.listdir(directory):
            old_path = os.path.join(directory, filename)
            if os.path.isfile(old_path):
                new_filename = filename.replace(' ', '_')
                new_path = os.path.join(directory, new_filename)
                if old_path != new_path:
                    os.rename(old_path, new_path)
                    print(f"Renamed: '{old_path}' -> '{new_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove spaces from filenames in a directory.")
    parser.add_argument('directory', type=str, help='Path to the directory containing files to rename.')
    parser.add_argument('--recursive', action='store_true', help='Recursively rename files in subdirectories.')
    args = parser.parse_args()

    remove_spaces_in_filenames(args.directory, args.recursive)

'''
Usage
python utils/noSpace.py /Users/alessandrofolloni/PycharmProjects/volleyball_tracker/videos/Converted
'''