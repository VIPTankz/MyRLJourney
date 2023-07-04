import pickle


class ChurnData:
    def __init__(self, avg_churn, per90, per99, per99_9, churns_per_action, percent_churns_per_action,
                 total_action_percents, churn_std, action_std, top50churns, game, start_timesteps, end_timesteps,
                 percent0churn, algo_name, median_churn):

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

    file = "churn_results\\BankHeist"

    files = [file + "25000_0.pkl", file + "75000_0.pkl"]

    for filename in files:

        with open(filename, 'rb') as inp:
            churn_data = pickle.load(inp)

            print("Churn Data for " + filename)
            print("Game: " + str(churn_data.game))
            print("Start Timestep: " + str(churn_data.start_timesteps))
            print("End Timestep: " + str(churn_data.end_timesteps))
            print("\n")
            print("Average Churn: " + str(churn_data.avg_churn))
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


