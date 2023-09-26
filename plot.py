import numpy as np
import matplotlib.pyplot as plt
runs = 5
smoothing = 20
"""
games = ["Alien","Amidar","Assault","Asterix","BankHeist","BattleZone","Boxing","Breakout","ChopperCommand","CrazyClimber",\
             "DemonAttack","Freeway","Frostbite","Gopher","Hero","Jamesbond","Kangaroo","Krull","KungFuMaster",\
             "MsPacman","Pong","PrivateEye","Qbert","RoadRunner","Seaquest","UpNDown"]
"""
games = ["Alien"]

for game in games:
    labels = ["DrQ","DrQOld"]
    data_files = []
    for i in labels:
        data_files.append(i + game + "Experiment")


    expers = []

    for exper in data_files:
        temp = []
        for i in range(runs):
            temp.append(np.load(exper + " (" + str(i) + ').npy'))
        expers.append(temp[:])

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    """fakeLists = []
    for i in range(runs):
        fakeList = []
        last_step = 100
        for i in range(200):
            fakeList.append([-np.random.random() * i + i,last_step])
            last_step += np.random.random() * 2000
            if last_step > 100000:
                break
        #experiement has form [score,steps] for every episode
        fakeList = np.array(fakeList)
        fakeLists.append(fakeList)

    fakeLists = np.array(fakeLists)
    print(fakeLists.shape)"""

    final_plot_points = []
    final_error_points = []

    for exper in range(len(data_files)):
        plot_points = []
        std_points = []
        for i in range(100):
            temp = []
            avgs = []
            for j in range(runs):
                idx = find_nearest(expers[exper][j][:,1],1000 * (i + 1))
                temp_list = expers[exper][j][:,0][:idx]
                avgs.append(np.mean(temp_list[-smoothing:]))

            plot_points.append(np.mean(avgs))
            std_points.append(np.std(avgs) / (runs ** 0.5))

        plot_points = np.array(plot_points)
        err_points = np.array(std_points) / np.sqrt(runs)

        final_plot_points.append(plot_points[:])
        final_error_points.append(err_points[:])

    final_error_points = np.array(final_error_points)
    final_plot_points = np.array(final_plot_points)

    plt.xlabel('Training Steps (1000s)')
    plt.ylabel('Score')
    plt.title('Scores During Training On Atari ' + game)

    for i in range(len(labels)):
        plt.plot(final_plot_points[i], label=labels[i])
        plt.fill_between(np.arange(100),final_plot_points[i] + final_error_points[i],
                         final_plot_points[i] - final_error_points[i], alpha=0.2)

    plt.legend(loc='upper left')
    plt.show()
