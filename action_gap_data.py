import numpy as np

games = ["Alien", "Amidar", "Assault", "Asterix", "BankHeist", "BattleZone", "Boxing", "Breakout", "ChopperCommand",
         "CrazyClimber", "DemonAttack", "Freeway", "Frostbite", "Gopher", "Hero", "Jamesbond", "Kangaroo", "Krull",
         "KungFuMaster", "MsPacman", "Pong", "PrivateEye", "Qbert", "RoadRunner", "Seaquest", "UpNDown"]

names = ["DDQN_n3"]
runs = 1

data_files = [[] for i in range(len(names))]
results_med = []
results_mean = []

results_med.append([])
results_mean.append([])
for i in range(len(names)):
    for game in games:
        data_files[i].append("action_gap_results\\" + names[i] + "\\" + names[i] + "_actionGaps" + game)

        print("\n" + game + " Action Gaps")

        for run in range(runs):
            print(np.median(np.load(data_files[i][-1] + str(run) + '.npy')))
            results_med[-1].append(np.median(np.load(data_files[i][-1] + str(run) + '.npy')))
            results_mean[-1].append(np.mean(np.load(data_files[i][-1] + str(run) + '.npy')))

for i in range(len(names)):
    print(results_med)
    print("\nAlgorithm: " + str(names[i]))
    print("Mean Action Gap: " + str(np.mean(results_mean[i])))
    print("Median Action Gap: " + str(np.mean(results_med[i])))