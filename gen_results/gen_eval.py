import matplotlib.pyplot as plt
import numpy as np

runs = 1
games = ["Breakout"] #["Alien", "Breakout", "Hero", "PrivateEye"]
algorithms = ["EffDQN", "EffDQN_lr_conv1e-5", "EffDQN_unique_mlps", "EffDQN_unique_conv_layers"]

labels = algorithms[:]

files = []

for run in range(runs):
    for game in games:
        for algorithm in algorithms:
            files.append(algorithm + "_" + game + str(run) + "_target_gen_data.npy")

data = []
for file in files:
    data.append(np.load(file))

plt.xlabel('Training Steps (1000s)')
plt.ylabel('Average % churn on target action')
plt.title('How much churn is from a targeted action?')

count = 0
for dat in data:
    plt.plot(dat, label=labels[count])
    count += 1

plt.legend(loc='upper right')
plt.show()

print("\nFinal Targeted Action %: ")
for i in range(len(data)):
    print(labels[i] + ": " + str(round(100 * data[i][-1], 2)))
