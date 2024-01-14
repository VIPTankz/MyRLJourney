import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd

games = ["bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun", "dodgeball", "fruitbot", "heist", "jumper",
         "leaper", "maze", "miner", "ninja", "plunder", "starpilot"]

games_to_include = [x.lower() for x in games]

# This is Using IQMs NOT Median-Normalised
#DrQ scores taken from BBF



human_scores = np.array([40, 13, 12, 13, 12.6, 10, 19, 32.4, 10, 10, 10, 10, 13, 10, 30, 64],dtype=np.float64)

random_scores = np.array([1, .5, 3.5, .5, 2, 5, 1.5, -1.5, 3.5, 3, 3, 5, 1.5, 3.5, 4.5, 2.5],dtype=np.float64)

#human_scores_hard = np.array([40, 13, 13.4, 14.2, 10, 19, 27.2, 10, 10],dtype=np.float64)

#random_scores_hard = np.array([0, .5, 2, .5, 5, 1.5, -.5, 2, 1],dtype=np.float64)


# Now, 'numpy_array' is a NumPy array with a shape of "runs" x "game" for the specified games


#to calc things, just get results in runs x game matrix and use the following:

#optimality gap
#return gamma - np.mean(np.minimum(scores, gamma))

#IQM
#return scipy.stats.trim_mean(scores, 0.25, axis=None)

#median
#return np.median(np.mean(scores, axis=-2, keepdims=keepdims), axis=-1)

#mean
#return np.mean(np.mean(scores, axis=-2, keepdims=keepdims), axis=-1)

#formula is ( (algo - human) / (human - random))

games = ["bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun", "dodgeball", "fruitbot", "heist", "jumper",
         "leaper", "maze", "miner", "ninja", "plunder", "starpilot"]

hns = []
labels = ["StableDQN_no_TR"]
runs = 5
expers = [[] for i in range(len(labels))]
data_files = [[] for i in range(len(labels))]
count = 0
for game in games:

    for i in range(len(labels)):
        data_files[i].append("procgen_results\\" + labels[i] + "\\" + labels[i] + game + "Evaluation")

        print("\n" + game + " Evaluation Scores")

        for run in range(runs):
            print(np.mean(np.load(data_files[i][-1] + " (" + str(run) + ').npy')))

# get data in for runs x games

#results = np.zeros((len(labels),len(games)),dtype=np.float64)
results = []
for i in range(len(labels)):
    for j in range(len(games)):
        x = []
        for run in range(runs):
            average_score = np.mean(np.load(data_files[i][j] + " (" + str(run) + ').npy'),axis=-1)
            human_normed = (average_score - random_scores[j]) / (human_scores[j] - random_scores[j])
            x.append(human_normed)
        results.append(x)

results = np.array(results).swapaxes(1, 0)
print(results.shape)

print("IQM:")
print(round(scipy.stats.trim_mean(results, 0.25, axis=None),3))

print("Optimality Gap:")
print(round(np.mean(np.minimum(results, 1.0)), 3))

print("Median:")
print(round(np.median(np.mean(results, axis=-2, keepdims=False), axis=-1), 3))

print("Mean:")
print(round(np.mean(np.mean(results, axis=-2, keepdims=False), axis=-1), 3))
