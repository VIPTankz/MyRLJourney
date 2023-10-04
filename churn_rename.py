import os

def rename_files_in_directory(directory_path, old_substrings, new_substrings):
    # Get a list of all files in the directory
    files = os.listdir(directory_path)

    for file_name in files:
        original_path = os.path.join(directory_path, file_name)

        # Check if the file name contains any of the old substrings
        renamed = False
        for old, new in zip(old_substrings, new_substrings):
            if old in file_name:
                new_file_name = file_name.replace(old, new)
                new_path = os.path.join(directory_path, new_file_name)
                os.rename(original_path, new_path)
                renamed = True
                break  # Stop searching if a match is found

        if renamed:
            print(f'Renamed: {file_name} => {new_file_name}')

# Example usage:
directory_path = "processing"  # Change this to your directory path
old_substrings = ["1600", "75000"]
new_substrings = ["ChurnEarly", "ChurnLate"]

rename_files_in_directory(directory_path, old_substrings, new_substrings)
