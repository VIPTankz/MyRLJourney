import matplotlib.pyplot as plt
import numpy as np
print("hi")
runs = 5
games = ["Alien", "Amidar", "Assault", "Asterix", "BankHeist", "BattleZone", "Boxing", "Breakout", "ChopperCommand",
         "CrazyClimber", \
         "DemonAttack", "Freeway", "Frostbite", "Gopher", "Hero", "Jamesbond", "Kangaroo", "Krull", "KungFuMaster", \
         "MsPacman", "Pong", "PrivateEye", "Qbert", "RoadRunner", "Seaquest", "UpNDown"]

algorithms = ["DDQN_n3"]
labels = ["n=10"]


def resize_arrays(array_list):
    # Find the length of the shortest array
    min_length = min(len(arr) for arr in array_list)

    # Resize all arrays to the same length
    resized_arrays = [arr[:min_length] if len(arr) > min_length else np.pad(arr, (0, min_length - len(arr)), 'constant')
                      for arr in array_list]

    return resized_arrays

def average_array(original_array):
    window_size = 1000
    result_array = np.empty_like(original_array, dtype=float)

    # Compute the rolling average and fill the result_array
    for i in range(len(original_array)):
        if i < window_size:
            # If there are not enough previous values, take the average of the available ones
            result_array[i] = np.mean(original_array[:i+1])
        else:
            # Compute the average of the previous 50 values
            result_array[i] = np.mean(original_array[i-window_size+1:i+1])

    return result_array

reward_data = []
bootstrap_data = []

for j in algorithms:
    game_temp_b = []
    game_temp_r = []
    for i in games:
        temp_b = []
        temp_r = []
        for run in range(runs):
            temp_r.append(np.load("proportion_results\\" + j + "\\" + j + "_proportionRewards" + i + str(run) + ".npy"))
            temp_b.append(np.load("proportion_results\\" + j + "\\" + j + "_proportionBootstrap" + i + str(run) + ".npy"))

        game_temp_b.append(temp_b)
        game_temp_r.append(temp_r)

    reward_data.append(game_temp_r)
    bootstrap_data.append(game_temp_b)

print(np.array(reward_data).shape)
reward_data = np.array(reward_data)
bootstrap_data = np.array(bootstrap_data)

reward_data = reward_data.mean(axis=2)
bootstrap_data = bootstrap_data.mean(axis=2)
print(np.array(reward_data).shape)


reward_data = resize_arrays(reward_data)
bootstrap_data = resize_arrays(bootstrap_data)

avg = [0 for i in range(len(algorithms))]

for i in range(len(games)):
    for j in range(len(algorithms)):

        y = np.array(reward_data[j][i]) / (reward_data[j][i] + bootstrap_data[j][i])
        x = np.arange(len(y)) / 1000
        y = average_array(y)
        plt.plot(x,y,label=labels[j])
        print(str(games[i]) + " Final Proportion: " + str(y[-1]))
        avg[j] += y[-1]

    plt.legend()
    plt.title(games[i])
    plt.xlabel("Timesteps (1000s)")
    plt.ylabel("Reward Proportion")
    #plt.show()

for i in range(len(algorithms)):
    print(str(algorithms[i]) + ": " + str(round(((avg[i] / len(games))*100),1)) + "%")
