import numpy as np
import matplotlib.pyplot as plt
import scipy


# This is Using IQMs NOT Median-Normalised
#DrQ scores taken from BBF

human_scores = np.array([7127.7, 1719.5, 742.0, 8503.3, 753.1, 37187.5, 12.1, 30.5, 7387.8,
                         35829.4, 1971.0, 29.6, 4334.7, 2412.5, 30826.4, 302.8, 3035.0, 2665.5,
                         22736.3, 6951.6, 14.6, 69571.3, 13455.0, 7845.0, 42054.7, 11693.2],dtype=np.float64)

random_scores = np.array([227.8, 5.8, 222.4, 210.0, 14.2, 2360.0, 0.1, 1.7, 811.0, 10780.5,
                          152.1, 0.0, 65.2, 257.6, 1027.0, 29.0, 52.0, 1598.0, 258.5, 307.3,
                          -20.7, 24.9, 163.9, 11.5, 68.4, 533.4],dtype=np.float64)


bbf_baseline = np.array([1173.2, 244.6, 2098.5, 3946.1, 732.9, 24459.8, 85.8, 370.6, 7549.3,
                         58431.8, 13341.4, 25.5, 2384.8, 1331.2, 7818.6, 1129.6, 6614.7, 8223.4,
                         18991.7, 2008.3, 16.7, 40.5, 4447.1, 33426.8, 1232.5, 12101.7],dtype=np.float64)

DrQ_baseline = np.array([771.2,102.8,452.4,603.5,168.9,12954,6,16.1,780.3,20516.5,1113.4,9.8,331.1,636.3,
                3736.3,236,940.6,4018.1,9111,960.5,-8.5,-13.6,854.4,8895.1,301.2,3180.8],dtype=np.float64)


#drq = np.divide(DrQ_baseline - human_scores, human_scores - random_scores)
#drq = np.divide(DrQ_baseline - random_scores, human_scores - random_scores)
print("BBF:")
bbf = (bbf_baseline - random_scores) / (human_scores - random_scores)
print(bbf)
print(scipy.stats.trim_mean(bbf, 0.25, axis=None))

print("DrQ(e)")
drq = np.divide(DrQ_baseline - random_scores, human_scores - random_scores)
print(scipy.stats.trim_mean(drq, 0.25, axis=None))

raise Exception("Stop")
#formula is ( (algo - human) / (human - random))


games = ["Alien","Amidar","Assault","Asterix","BankHeist","BattleZone","Boxing","Breakout","ChopperCommand","CrazyClimber",\
             "DemonAttack","Freeway","Frostbite","Gopher","Hero","Jamesbond","Kangaroo","Krull","KungFuMaster",\
             "MsPacman","Pong","PrivateEye","Qbert","RoadRunner","Seaquest","UpNDown"]

hns = []
labels = ["DER"]
runs = 5
expers = [[] for i in range(len(labels))]
data_files = [[] for i in range(len(labels))]
count = 0
for game in games:

    for i in range(len(labels)):
        data_files[i].append("results\\" + labels[i] + "\\" + labels[i] + game + "Evaluation")

        print("\n" + game + " Evaluation Scores")

        for run in range(runs):
            print(np.mean(np.load(data_files[i][-1] + " (" + str(run) + ').npy')))

#print(data_files)

results = np.zeros((len(labels),len(games)),dtype=np.float64)
for i in range(len(labels)):
    for j in range(len(games)):
        x = []
        for run in range(runs):
            x.append(np.mean(np.load(data_files[i][j] + " (" + str(run) + ').npy')))
        results[i, j] = sum(x) / len(x)
