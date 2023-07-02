import numpy as np
import statistics
np.set_printoptions(suppress=True)
runs = 5


human_scores = np.array([7127.80,1719.53,742.00,8503.33,753.13,\
                37187.50,12.06,30.47,7387.80,35829.41,\
                1971.00,29.60,4334.67,2412.50,30826.38,\
                302.80,3035.00,2665.53,22736.25,6951.60,\
                14.59,69571.27,13455.00,7845.00,42054.71,11693.23],dtype=np.float64)
print(len(human_scores))

random_scores = np.array([227.8,5.8,222.4,210.0,14.2,2360.0,0.1,1.7,811,10780.5,
                 152.1,0.0,65.2,257.6,1027,29,52,1598,258.5,307.3,-20.7,24.9,
                 163.9,11.5,68.4,533.4],dtype=np.float64)
print(len(random_scores))
human_scores = np.subtract(human_scores,random_scores)


DrQ_baseline = np.array([771.2,102.8,452.4,603.5,168.9,12954,6,16.1,780.3,20516.5,1113.4,9.8,331.1,636.3,
                3736.3,236,940.6,4018.1,9111,960.5,-8.5,-13.6,854.4,8895.1,301.2,3180.8],dtype=np.float64)

x = np.divide(DrQ_baseline - random_scores,human_scores)
x = np.median(x)
print(x)

#"Breakout","Pong",
games = ["Alien","Amidar","Assault","Asterix","BankHeist","BattleZone","Boxing","Breakout","ChopperCommand","CrazyClimber",\
             "DemonAttack","Freeway","Frostbite","Gopher","Hero","Jamesbond","Kangaroo","Krull","KungFuMaster",\
             "MsPacman","Pong","PrivateEye","Qbert","RoadRunner","Seaquest","UpNDown"]

print("HERE")
print(np.load('DrQ\\DrQAlienEvaluation (0).npy'))
#games = ["Alien"]

print_ind = True

print(len(games))

hns = []
labels = ["DrQ"]
expers = [[] for i in range(len(labels))]
data_files = [[] for i in range(len(labels))]
#games = ["Seaquest","UpNDown"]
count = 0
for game in games:


    for i in range(len(labels)):
        data_files[i].append(labels[i] + "\\" + labels[i] + game + "Evaluation")

    print("\n" + game + " Evaluation Scores")
    print(np.load(data_files[i][-1] + " (" + str(i) + ').npy'))

print(data_files)

results = np.zeros((len(labels),len(games)),dtype=np.float64)
for i in range(len(labels)):
    for j in range(len(games)):
        results[i, j] = np.mean(np.load(data_files[i][j] + " (" + str(i) + ').npy'))

print(results)
results = np.subtract(results, random_scores)
results = np.divide(results, human_scores)
print(results)
print("Human-Normalised Medians:")
print(np.median(results,axis=1))


    
