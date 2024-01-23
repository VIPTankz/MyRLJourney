import numpy as np
import matplotlib.pyplot as plt
runs = 5
smoothing = 250

games = ["Alien","Amidar","Assault","Asterix","BankHeist","BattleZone","Boxing","Breakout","ChopperCommand","CrazyClimber",\
             "DemonAttack","Freeway","Frostbite","Gopher","Hero","Jamesbond","Kangaroo","Krull","KungFuMaster",\
             "MsPacman","Pong","PrivateEye","Qbert","RoadRunner","Seaquest","UpNDown"]

#games = ["CartPole"]

"""games = ["bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun", "dodgeball", "fruitbot", "heist", "jumper",
         "leaper", "maze", "miner", "ninja", "plunder", "starpilot"]

games = ["coinrun"]"""
#games = ["Battlezone", "Hero"]


def find_nearest(array, value):
    array = np.asarray(array)
    idxs = (np.abs(array - value)).argmin()
    return idxs

#fig, axs = plt.subplots(1, 1, figsize=(15, 10))
#axs = axs.flatten()

for idx, game in enumerate(games):
    labels = ["StableDQN"]
    data_files = []
    for i in labels:
        data_files.append(i + "\\" + i + game + "Experiment")

    expers = []

    for exper in data_files:
        temp = []
        for i in range(runs):
            temp.append(np.load("results\\" + exper + " (" + str(i) + ').npy'))
        expers.append(temp[:])

    final_plot_points = []
    final_error_points = []

    for exper in range(len(data_files)):
        plot_points = []
        std_points = []
        for i in range(100):
            temp = []
            avgs = []
            for j in range(runs):
                idxz = find_nearest(expers[exper][j][:,1],1000 * (i + 1))
                temp_list = expers[exper][j][:,0][:idxz]
                avgs.append(np.mean(temp_list[-smoothing:]))

            plot_points.append(np.mean(avgs))
            std_points.append(np.std(avgs) / (runs ** 0.5))

        plot_points = np.array(plot_points)
        err_points = np.array(std_points) / np.sqrt(runs)

        final_plot_points.append(plot_points[:])
        final_error_points.append(err_points[:])

    final_error_points = np.array(final_error_points)
    final_plot_points = np.array(final_plot_points)

    plt.xlabel('Training Steps (1000s)', fontsize=16) #[idx]
    plt.ylabel('Score', fontsize=16) #[idx]
    plt.title(game) #[idx]

    plt.tick_params(axis='x', labelsize=14)  # Larger x-axis tick labels #[idx]
    plt.tick_params(axis='y', labelsize=14) #[idx]

    for i in range(len(labels)):
        if i == 1:
            plt.plot(final_plot_points[i], label="DQN") #[idx]
        else:
            plt.plot(final_plot_points[i], label=labels[i]) #[idx]
        plt.fill_between(np.arange(100), final_plot_points[i] + final_error_points[i],
                              final_plot_points[i] - final_error_points[i], alpha=0.2) #[idx]

    plt.legend(loc='upper left') #[idx]

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
