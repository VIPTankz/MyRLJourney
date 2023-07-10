import pickle


class ChurnData:
    def __init__(self, avg_churn, per90, per99, per99_9, churns_per_action, percent_churns_per_action,
                 total_action_percents, churn_std, action_std, top50churns, game, start_timesteps, end_timesteps,
                 percent0churn, algo_name, median_churn): #

        self.avg_churn = avg_churn
        self.median_churn = median_churn
        self.per90 = per90
        self.per99 = per99
        self.per99_9 = per99_9
        self.churns_per_action = churns_per_action
        self.percent_churns_per_action = percent_churns_per_action
        self.total_action_percents = total_action_percents
        self.churn_std = churn_std
        self.action_std = action_std
        self.top50churns = top50churns
        self.game = game
        self.start_timesteps = start_timesteps
        self.end_timesteps = end_timesteps
        self.percent0churn = percent0churn
        self.algo_name = algo_name

if __name__ == "__main__":


    games = ["Alien","Amidar","Assault","Asterix","BankHeist","BattleZone","Boxing","Breakout","ChopperCommand","CrazyClimber",\
             "DemonAttack","Freeway","Frostbite","Gopher","Hero","Jamesbond","Kangaroo","Krull","KungFuMaster",\
             "MsPacman","Pong","PrivateEye","Qbert","RoadRunner","Seaquest","UpNDown"]

    games = ["Alien","Amidar","Assault","Asterix","BankHeist","Breakout","ChopperCommand","CrazyClimber",\
             "DemonAttack","Freeway","Hero","Jamesbond","Kangaroo","Krull","KungFuMaster",\
             "PrivateEye","Qbert","RoadRunner","Seaquest","UpNDown"]

    churn_25 = 0
    churn_75 = 0
    churn_std25 = 0
    churn_std75 = 0
    action_std25 = 0
    action_std75 = 0


    for game in games:
        file_ = ["churn_results\\DER_noiseless\\DER_noiseless_churn_" + game]

        files = []
        for i in file_:
            files.append(i + "1600_0.pkl")
            files.append(i + "75000_0.pkl")

        for filename in files:
            print("---------------------------------------")
            with open(filename, 'rb') as inp:
                churn_data = pickle.load(inp)

                print("Churn Data for " + filename)
                print("Game: " + str(churn_data.game))
                print("Start Timestep: " + str(churn_data.start_timesteps))
                print("End Timestep: " + str(churn_data.end_timesteps))
                print("\n")
                print("Average Churn: " + str(churn_data.avg_churn))
                print("Median Churn: " + str(churn_data.median_churn))
                print("90th Percentile: " + str(churn_data.per90))
                print("99th Percentile: " + str(churn_data.per99))
                print("99.9th Percentile: " + str(churn_data.per99_9))
                print("Percent with 0 Churn: " + str(churn_data.percent0churn))
                print("Top 50 Churns: " + str(['%.3f' % elem for elem in churn_data.top50churns]))
                print("\n")
                print("Total Churn per Action  : " + str(['%.5f' % elem for elem in churn_data.churns_per_action]))
                print("Percent Churn per Action: " + str(['%.5f' % elem for elem in churn_data.percent_churns_per_action]))
                print("Total Action Percent    : " + str(['%.5f' % elem for elem in churn_data.total_action_percents]))
                print("Churn std: " + str(round(churn_data.churn_std, 5)))
                print("Action std: " + str(round(churn_data.action_std, 5)))
                print("\n\n")

                if churn_data.start_timesteps == 1600:
                    churn_25 += churn_data.avg_churn
                    churn_std25 += churn_data.churn_std
                    action_std25 += churn_data.action_std
                else:
                    churn_75 += churn_data.avg_churn
                    churn_std75 += churn_data.churn_std
                    action_std75 += churn_data.action_std

    print("================")
    print("AVG churn early: " + str(churn_25 / len(games)))
    print("AVG churn late: " + str(churn_75 / len(games)))
    print("AVG churn std early: " + str(churn_std25 / len(games)))
    print("AVG churn std late: " + str(churn_std75 / len(games)))
    print("AVG action std early: " + str(action_std25 / len(games)))
    print("AVG action std late: " + str(action_std75 / len(games)))

