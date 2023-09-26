import os

# Specify the directory where your files are located
directory_path = "C:\\Users\\TYLER\\Documents\\GitHub\\MyRLJourney\\churn_results\\DDQN"


# Function to find a suitable new run number
def find_new_run_number(gamename, existing_filenames):
    new_run_number = 0
    new_filename = f"DDQN_{gamename}_{new_run_number}.pkl"

    while new_filename in existing_filenames:
        new_run_number += 1
        new_filename = f"DDQN_{gamename}_{new_run_number}.pkl"

    return new_run_number, new_filename


# Get a list of existing filenames in the directory
existing_filenames = set(os.listdir(directory_path))

# Iterate through all files in the directory
for filename in os.listdir(directory_path):
    # Split the filename into parts using underscores
    parts = filename.split("_")

    # Check if the filename has the expected format
    if len(parts) == 3 and filename.startswith("DDQN") and filename.endswith(".pkl"):
        gamename = parts[1]
        run_number = int(parts[2].split(".pkl")[0])

        # Find a suitable new run number
        new_run_number, new_filename = find_new_run_number(gamename, existing_filenames)

        # Rename the file
        os.rename(os.path.join(directory_path, filename), os.path.join(directory_path, new_filename))
        existing_filenames.remove(filename)  # Remove the old filename from the set
        existing_filenames.add(new_filename)  # Add the new filename to the set

        print(f"Renamed: {filename} -> {new_filename}")
