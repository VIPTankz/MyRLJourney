import os
import pickle
import numpy as np


def save_files_to_existing_pickle(directory_path, pickle_filename):
    file_data = {}

    # Check if the pickle file already exists in the "whole_results" subdirectory
    result_pickle_path = os.path.join("whole_results", pickle_filename)
    if os.path.exists(result_pickle_path):
        with open(result_pickle_path, 'rb') as existing_pickle:
            file_data = pickle.load(existing_pickle)

    for filename in os.listdir(directory_path):
        if filename.endswith('.pkl'):
            with open(os.path.join(directory_path, filename), 'rb') as file:
                file_data[os.path.splitext(filename)[0]] = pickle.load(file)
        elif filename.endswith('.npy'):
            file_data[os.path.splitext(filename)[0]] = np.load(os.path.join(directory_path, filename))

    # Save the pickle file inside the "whole_results" subdirectory
    with open(result_pickle_path, 'wb') as pickle_file:
        pickle.dump(file_data, pickle_file)

if __name__ == "__main__":
    directory_path = "processing"  # Change this to your directory path
    pickle_filename = "DDQN_n1.RESULT"  # Change this to the desired pickle filename
    save_files_to_existing_pickle(directory_path, pickle_filename)

    print("Done!")

