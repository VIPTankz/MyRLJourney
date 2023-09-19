import matplotlib.pyplot as plt
import numpy as np

games = ["Breakout"]
algorithms = ["DDDQN_n10_proportion"]
labels = ["n=10", "n=1"]

reward_data = []
bootstrap_data = []

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

for j in algorithms:
    game_temp_b = []
    game_temp_r = []
    for i in games:
        game_temp_r.append(np.load(j + "Rewards" + i + ".npy"))
        game_temp_b.append(np.load(j + "Bootstrap" + i + ".npy"))

    reward_data.append(game_temp_r)
    bootstrap_data.append(game_temp_b)

for i in range(len(games)):
    for j in range(len(algorithms)):
        x = np.arange(len(reward_data[0][0])) / 1000
        y = np.array(reward_data[j][i]) / (reward_data[j][i] + bootstrap_data[j][i])
        y = average_array(y)
        plt.plot(x,y,label=labels[j])

    plt.legend()
    plt.title(games[i])
    plt.xlabel("Timesteps (1000s)")
    plt.ylabel("Reward Proportion")
    plt.show()
