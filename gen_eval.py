import matplotlib.pyplot as plt
import numpy as np

runs = 5
games = ["Alien", "Amidar", "Assault", "Asterix", "BankHeist", "BattleZone", "Boxing", "Breakout", "ChopperCommand",
         "CrazyClimber", "DemonAttack", "Freeway", "Frostbite", "Gopher", "Hero", "Jamesbond", "Kangaroo", "Krull",
         "KungFuMaster", "MsPacman", "Pong", "PrivateEye", "Qbert", "RoadRunner", "Seaquest", "UpNDown"]

algorithms = ["DDQN", "DDQN_split","DDQN_legged"]

labels = ["DDQN", "DDQN_MLPs","DDQN_CONVs"]

files = []

""" CODE FOR EACH GAME
for algorithm in algorithms:
    for game in games:
        temp = []
        for run in range(runs):
            temp.append("gen_results\\" + algorithm + "\\" + algorithm + "_" + game + str(run) + "_target_gen_data.npy")

        files.append(temp)

data = []
count = 0
for file in files:
    temp = []
    for i in file:
        temp.append(np.load(i))

    data.append(np.mean(np.array(temp),axis=0))
    print(games[count])
    print(data[-1][-1])
    count += 1


data = np.array(data)
print(data.shape)
plt.xlabel('Training Steps')
plt.ylabel('Average % churn on target action')

count = 0
for dat in data:
    plt.plot(dat, label=games[count], alpha=0.4, color="grey")
    count += 1

#plot average
data = np.mean(data, axis=0)
plt.plot(dat, color="blue", linewidth=3)

print("\nFinal Targeted Action %: ")
print(dat[-1] * 100)
plt.show()
"""

#CODE FOR COMPARING ALGOS
for algorithm in algorithms:
    temp0 = []
    for game in games:
        temp = []
        for run in range(runs):
            try:
                _ = np.load("gen_results\\" + algorithm + "\\" + algorithm + "_" + game + str(run) + "_target_gen_data.npy")
                temp.append("gen_results\\" + algorithm + "\\" + algorithm + "_" + game + str(run) + "_target_gen_data.npy")

            except:
                temp.append(
                    "gen_results\\" + algorithm + "\\" + algorithm + "_" + game + str(0) + "_target_gen_data.npy")

        temp0.append(temp)
    files.append(temp0)

data = []

for algo in files:
    count = 0
    for file in algo:
        temp = []
        for i in file:
            temp.append(np.load(i))

        data.append(np.mean(np.array(temp),axis=0))
        print(games[count])
        print(data[-1][-1])
        count += 1


data = np.array(data)
data = np.array(np.split(data, len(algorithms)))
print(data.shape)
plt.xlabel('Training Steps')
plt.ylabel('Average % churn on target action')

data = data * 100


#plot average
count = 0
for dat in data:
    dat = np.mean(dat, axis=0)
    plt.plot(dat, label=labels[count])
    count += 1

print("\nFinal Targeted Action %: ")
print(dat[-1] * 100)

plt.legend()
plt.show()


