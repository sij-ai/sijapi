import os
import re
from pathlib import Path

# Set the maximum permissible file name length for NextCloud
MAX_FILENAME_LENGTH = 255

# Define impermissible characters for NextCloud
IMPERMISSIBLE_CHARS = r'[<>:"/\\|?*\n]'

def sanitize_file_name(file_name):
    """Sanitize the file name by replacing impermissible characters and truncating if necessary."""
    # Replace impermissible characters with underscores
    sanitized_name = re.sub(IMPERMISSIBLE_CHARS, '_', file_name)
    # Truncate the file name if it exceeds the maximum length
    if len(sanitized_name) > MAX_FILENAME_LENGTH:
        ext = Path(sanitized_name).suffix
        base_name = sanitized_name[:MAX_FILENAME_LENGTH - len(ext)]
        sanitized_name = base_name + ext
    return sanitized_name

def check_file_name(file_name):
    """Check if the file name is impermissibly long or contains impermissible characters."""
    if len(file_name) > MAX_FILENAME_LENGTH:
        return True
    if re.search(IMPERMISSIBLE_CHARS, file_name):
        return True
    return False

def list_and_correct_impermissible_files(root_dir, rename: bool = False):
    """List and correct all files with impermissible names."""
    impermissible_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if check_file_name(filename):
                file_path = Path(dirpath) / filename
                impermissible_files.append(file_path)
                print(f"Impermissible file found: {file_path}")

                # Sanitize the file name
                new_filename = sanitize_file_name(filename)
                new_file_path = Path(dirpath) / new_filename

                # Ensure the new file name does not already exist
                if new_file_path.exists():
                    counter = 1
                    base_name, ext = os.path.splitext(new_filename)
                    while new_file_path.exists():
                        new_filename = f"{base_name}_{counter}{ext}"
                        new_file_path = Path(dirpath) / new_filename
                        counter += 1

                # Rename the file
                if rename == True:
                    os.rename(file_path, new_file_path)
                    print(f"Renamed: {file_path} -> {new_file_path}")

    return impermissible_files

def process_nc(dir_to_fix, rename: bool = False):
    impermissible_files = list_and_correct_impermissible_files(dir_to_fix, rename)
    if impermissible_files:
        print("\nList of impermissible files found and corrected:")
        for file in impermissible_files:
            print(file)
    else:
        print("No impermissible files found.")
