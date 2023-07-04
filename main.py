import gym
import numpy as np
from gym.wrappers import AtariPreprocessing
import time
from copy import deepcopy
import sys
import torch as T

if __name__ == '__main__':

    from DrQ_Sep_Agent import Agent
    agent_name = "DrQ_Sep"

    """
    games = ["Alien","Amidar","Assault","Asterix","BankHeist","BattleZone","Boxing","Breakout","ChopperCommand","CrazyClimber",\
             "DemonAttack","Freeway","Frostbite","Gopher","Hero","Jamesbond","Kangaroo","Krull","KungFuMaster",\
             "MsPacman","Pong","PrivateEye","Qbert","RoadRunner","Seaquest","UpNDown"]
    """

    """
    gameset = [["Alien","Amidar","Assault","Asterix"],["BankHeist","BattleZone","Boxing","Breakout"],
               ["ChopperCommand","CrazyClimber","DemonAttack","Freeway"],["Frostbite","Gopher","Hero","Jamesbond"],
               ["Kangaroo","Krull","KungFuMaster","MsPacman"],["Pong","PrivateEye","Qbert"],["RoadRunner","Seaquest","UpNDown"]]"""

    gameset = [["Alien", "Amidar", "Assault", "Asterix", "BankHeist", "BattleZone", "Boxing"],
               ["Breakout", "ChopperCommand", "CrazyClimber", "DemonAttack", "Freeway", "Frostbite", "Gopher"],
               ["Hero", "Jamesbond", "Kangaroo", "Krull", "KungFuMaster", "MsPacman", "Pong"],
               ["PrivateEye", "Qbert", "RoadRunner", "Seaquest", "UpNDown"]]

    gameset_idx = int(sys.argv[1])

    games = gameset[gameset_idx]
    print("Currently Playing Games: " + str(games))

    gpu = sys.argv[2]
    device = T.device('cuda:' + gpu if T.cuda.is_available() else 'cpu')
    print("Device: " + str(device))

    for game in games:
        for runs in range(1):
            env = gym.make('ALE/' + game + '-v5')
            env = AtariPreprocessing(env, frame_skip=1, terminal_on_life_loss=True)
            env = gym.wrappers.FrameStack(env, 4)
            env.seed(runs)
            print(env.observation_space)
            print(env.action_space)

            agent = Agent(n_actions=env.action_space.n, input_dims=[4, 84, 84], total_frames=100000, device=device,
                          game=game, run=runs)

            scores = []
            scores_temp = []
            n_steps = 100000
            steps = 0
            episodes = 0
            start = time.time()
            while steps < n_steps:

                score = 0
                episodes += 1
                done = False
                trun = False
                observation = env.reset()
                while not done and not trun:
                    steps += 1
                    action = agent.choose_action(observation)
                    observation_, reward, done, info = env.step(action)
                    score += reward
                    reward = np.clip(reward, -1., 1.)

                    agent.store_transition(observation, action, reward,
                                                  observation_ , done)

                    agent.learn()

                    observation = deepcopy(observation_)
                scores.append([score, steps])
                scores_temp.append(score)

                avg_score = np.mean(scores_temp[-50:])

                if episodes % 1 == 0:
                    print('{} {} avg score {:.2f} total_steps {:.0f} fps {:.2f}'
                          .format(agent_name, game, avg_score, steps, steps / (time.time() - start)), flush=True)

            fname = agent_name + game + "Experiment (" + str(runs) + ').npy'
            np.save(fname, np.array(scores))
            agent.set_eval_mode()
            evals = []
            steps = 0
            episodes = 0
            while episodes < 100:
                done = False
                trun = False
                observation = env.reset()
                score = 0
                while not done and not trun:
                    steps += 1
                    action = agent.choose_action(observation)
                    observation_, reward, done, info = env.step(action)
                    score += reward
                    observation = observation_

                evals.append(score)
                print("Evaluation Score: " + str(score))
                episodes += 1

            fname = agent_name + game + "Evaluation (" + str(runs) + ').npy'
            np.save(fname, np.array(evals))
