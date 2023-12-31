import matplotlib.pyplot as plt
import numpy as np
print("hi")
runs = 1
games = ["Alien", "Amidar", "Assault", "Asterix", "BankHeist", "BattleZone", "Boxing", "Breakout", "ChopperCommand",
         "CrazyClimber", \
         "DemonAttack", "Freeway", "Frostbite", "Gopher", "Hero", "Jamesbond", "Kangaroo", "Krull", "KungFuMaster", \
         "MsPacman", "Pong", "PrivateEye", "Qbert", "RoadRunner", "Seaquest", "UpNDown"]

algorithms = ["DDQN"]
labels = ["n=10", "n=3","n=1","n=1 gamma=0.9","n=10 gamma=0.999"]


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
        game_temp_r.append(np.load(j + "\\" + j + "_proportionRewards" + i + "0.npy"))
        game_temp_b.append(np.load(j + "\\" + j + "_proportionBootstrap" + i + "0.npy"))

    reward_data.append(game_temp_r)
    bootstrap_data.append(game_temp_b)

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
    print(str(algorithms[i]) + ": " + str(avg[i] / len(games)))
