import numpy as np
import statistics
runs = 5


human_scores = [7127.80,1719.53,742.00,8503.33,753.13,\
                37187.50,12.06,30.47,7387.80,35829.41,\
                1971.00,29.60,4334.67,2412.50,30826.38,\
                302.80,3035.00,2665.53,22736.25,6951.60,\
                14.59,69571.27,13455.00,7845.00,42054.71,11693.23]
print(len(human_scores))

#"Breakout","Pong",
games = ["Alien","Amidar","Assault","Asterix","BankHeist","BattleZone","Boxing","Breakout","ChopperCommand","CrazyClimber",\
             "DemonAttack","Freeway","Frostbite","Gopher","Hero","Jamesbond","Kangaroo","Krull","KungFuMaster",\
             "MsPacman","Pong","PrivateEye","Qbert","RoadRunner","Seaquest","UpNDown"]

games = ["Alien"]

print_ind = True

print(len(games))

hns = []
#games = ["Seaquest","UpNDown"]
count = 0
for game in games:
    labels = ["SPR"]
    data_files = []
    for i in labels:
        data_files.append(i + "\\" + i + game + "Evaluation")
    #data_files = ["DrQ" + game + "Evaluation"]
    print("\n" + game + " Evaluation Scores")

    for i in range(len(labels)):
        hns.append([])
    
    expers = []
    for exper in data_files:
        temp = []
        for i in range(runs):
            temp.append(np.mean(np.load(exper + " (" + str(i) + ').npy')))

        if print_ind:
            print(temp[:])
        expers.append(np.mean(temp[:]))

    expers = np.array(expers)


    for i in range(len(labels)):
        print(labels[i] + ": " + str(expers[i]))

        hns_ = (expers[i] / human_scores[count])
        hns[i].append(hns_)

    count += 1


print("\nHuman Normalized Scores")
for i in range(len(labels)):
    print(labels[i] + ": " + str(statistics.median(hns[i])))



    
