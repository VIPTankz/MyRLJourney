import numpy as np
import matplotlib.pyplot as plt


def moving_average(input_list, x):
    if x <= 0:
        raise ValueError("The window size (x) must be a positive integer.")

    moving_averages = []
    for i in range(len(input_list) - x + 1):
        window = input_list[i:i + x]
        average = sum(window) / x
        moving_averages.append(average)

    return moving_averages


# this is for asterix only
games = ["Alien", "Amidar", "Assault", "Asterix", "BankHeist", "BattleZone", "Boxing", "Breakout", "ChopperCommand",
         "CrazyClimber", \
         "DemonAttack", "Freeway", "Frostbite", "Gopher", "Hero", "Jamesbond", "Kangaroo", "Krull", "KungFuMaster", \
         "MsPacman", "Pong", "PrivateEye", "Qbert", "RoadRunner", "Seaquest", "UpNDown"]

game = "Asterix"
averaging = 500

names = ["DDQN_n3_discount95"]
data = []


for i in names:
    data.append(np.load("variance_results\\" + i + "\\" + i + "_varianceData" + game + "0.npy"))

print(data)
for i in range(len(data)):
    data[i] = moving_average(data[i], averaging)

for i in range(len(data)):
    plt.plot(data[i], label=names[i])

plt.xlabel("Steps")
plt.ylabel("Loss Variance")
plt.legend()
plt.show()
