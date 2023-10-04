import pickle
import numpy as np
import re


def open_pickle_file(pickle_filename):
    with open(pickle_filename, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    return data


def remove_substrings(substrings, target_string):
    while True:
        modified = False  # Flag to track if any substring was removed in this iteration
        for substring in substrings:
            if substring in target_string:
                target_string = target_string.replace(substring, '', 1)  # Remove only the first occurrence
                modified = True
        if not modified:
            break  # Exit the loop if no modifications were made in this iteration
    return target_string


def find_final_number(input_string):
    # Use a regular expression to find the final number in the string
    # \d+ matches one or more digits, \D* matches zero or more non-digits, $ matches the end of the string
    match = re.search(r'(\d+)\D*$', input_string)

    if match:
        return int(match.group(1))  # Convert the matched string to an integer
    else:
        return None  # Return None if no number is found

def extract_game_name(input_string, game_names):
    # Loop through the list of game names and search for each in the input string
    for game in game_names:
        if game in input_string:
            return game
    return None  # Return None if no game name is found



def main(algorithms, data_request):
    data = []



    games = ["Alien", "Amidar", "Assault", "Asterix", "BankHeist", "BattleZone", "Boxing", "Breakout", "ChopperCommand",
             "CrazyClimber", \
             "DemonAttack", "Freeway", "Frostbite", "Gopher", "Hero", "Jamesbond", "Kangaroo", "Krull", "KungFuMaster", \
             "MsPacman", "Pong", "PrivateEye", "Qbert", "RoadRunner", "Seaquest", "UpNDown"]

    remove_strings = []

    for i in algorithms:
        remove_strings.append(i + "_")

    for i in algorithms:
        remove_strings.append(i)

    for i in range(100):
        i = 99 - i
        remove_strings.append("(" + str(i) + ")")
        remove_strings.append("_" + str(i))

        for j in games:
            remove_strings.append(j + str(i))

    for i in games:
        remove_strings.append(i)

    remove_strings.sort(key=len, reverse=True)

    for i in range(len(algorithms)):
        pickle_filename = "whole_results/" + algorithms[i] + ".RESULT"  # Adjust the path as needed
        algo_files = open_pickle_file(pickle_filename)

        data_temp = {}

        for key, value in algo_files.items():
            # Depending on the data structure, you can print or process the loaded data here
            # For example, if the data is a numpy array, you can print it like this:

            removed_string = remove_substrings(remove_strings, key)

            if removed_string in data_temp:
                data_temp[removed_string].append([key, value])
            else:
                data_temp[removed_string] = [[key, value]]

        data.append(data_temp)


    #data is a list of dictionaries per algorithm
    # each dict has structure {"Shortened_form": [[filename, data], [filename, data],...]}

    data_counts = []

    print("Data For These algorithms:")
    count = 0
    for my_dict in data:
        keys = my_dict.keys()

        runs_temp = {}
        run_nums_per_cat = {}

        for key in keys:
            #key is the shortened part I constructed
            #value is [name of file, data]
            file_names = []
            for filename_data in my_dict[key]:
                file_names.append(filename_data[0])

            highest_run_numbers = {}

            for file_name in file_names:
                game_name = extract_game_name(file_name, games)
                run_number = find_final_number(file_name)
                if game_name in highest_run_numbers:
                    highest_run_numbers[game_name] = max(highest_run_numbers[game_name], run_number)
                else:
                    highest_run_numbers[game_name] = run_number

            lowest = 999
            for keyX, value in highest_run_numbers.items():
                if value < lowest:
                    lowest = value

            run_nums_per_cat[key] = lowest

        runs_temp[algorithms[count]] = run_nums_per_cat

        data_counts.append(runs_temp)
        count += 1

    #add one to the counts for each data as we start from 0
    for algo in data_counts:
        for key, value in algo.items():
            for key1, value1 in value.items():
                algo[key][key1] = value1 + 1

    print(data_counts)

    final_data = []
    for algo in range(len(algorithms)):
        temp = []
        data[algo][request]



if __name__ == "__main__":
    algorithms = ["DDQN", "DDQN_n1"]
    request = "Evaluation"
    main(algorithms, request)