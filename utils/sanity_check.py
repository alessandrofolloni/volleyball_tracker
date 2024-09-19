import os


def count_files_in_folder(folder_path):
    # List all entries in the directory
    entries = os.listdir(folder_path)

    # Count files (exclude directories)
    file_count = len([entry for entry in entries if os.path.isfile(os.path.join(folder_path, entry))])

    return file_count


# Example usage
folder_path = '/Users/alessandrofolloni/PycharmProjects/volleyball_tracker/videos/Converted'
print(f"Number of files in the folder: {count_files_in_folder(folder_path)}")

folder_path2 = '/Users/alessandrofolloni/PycharmProjects/volleyball_tracker/videos/original'
print(f"Number of files in the folder: {count_files_in_folder(folder_path2)}")