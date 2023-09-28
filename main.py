import gym
import numpy as np
from gym.wrappers import AtariPreprocessing
import time
from copy import deepcopy
import sys
import torch as T
from AtariSetup import AtariPreprocessing, TimeLimit, FrameStack, ImageToPyTorch


def make_env(game, eval):
    env = gym.make('ALE/' + game + '-v5')
    env.seed(runs + eval * 10000)

    env = AtariPreprocessing(env.env,
                             frame_skip=4,
                             max_random_noops=30,
                             terminal_on_life_loss=True)
    env = TimeLimit(env, max_episode_steps=108000)
    env = FrameStack(env, k=4)
    env = ImageToPyTorch(env)

    return env


if __name__ == '__main__':

    from DrQ_Agent_hacked import Agent

    agent_name = "DDQN"

    """
    games = ["Alien","Amidar","Assault","Asterix","BankHeist","BattleZone","Boxing","Breakout","ChopperCommand","CrazyClimber",\
             "DemonAttack","Freeway","Frostbite","Gopher","Hero","Jamesbond","Kangaroo","Krull","KungFuMaster",\
             "MsPacman","Pong","PrivateEye","Qbert","RoadRunner","Seaquest","UpNDown"]
    """

    # 6 sets - Iridis
    gameset = [["Alien","Amidar","Assault","Asterix"], ["BankHeist","BattleZone","Boxing","Breakout"],
               ["ChopperCommand","CrazyClimber","DemonAttack","Freeway"], ["Frostbite","Gopher","Hero","Jamesbond"],
               ["Kangaroo","Krull","KungFuMaster","MsPacman", "Pong"], ["PrivateEye", "Qbert", "RoadRunner", "Seaquest", "UpNDown"]]

    # 3 Sets - RTX 4090
    gameset = [["Alien","Amidar","Assault","Asterix", "BankHeist","BattleZone","Boxing","Breakout"],
               ["ChopperCommand","CrazyClimber","DemonAttack","Freeway", "Frostbite","Gopher","Hero","Jamesbond", "Kangaroo"],
               ["Krull","KungFuMaster","MsPacman", "Pong", "PrivateEye", "Qbert", "RoadRunner", "Seaquest", "UpNDown"]]


    #single set - util
    gameset = [["Alien","Amidar","Assault","Asterix","BankHeist","BattleZone","Boxing","Breakout",
                "ChopperCommand","CrazyClimber","DemonAttack","Freeway","Frostbite","Gopher","Hero","Jamesbond",
               "Kangaroo","Krull","KungFuMaster","MsPacman", "Pong","PrivateEye", "Qbert", "RoadRunner", "Seaquest", "UpNDown"]]


    gameset_idx = int(sys.argv[1])

    games = gameset[gameset_idx]
    print("Currently Playing Games: " + str(games))

    gpu = sys.argv[2]
    device = T.device('cuda:' + gpu if T.cuda.is_available() else 'cpu')
    print("Device: " + str(device))

    try:
        run = int(sys.argv[3])
        run_spec = True
        print("Run number: " + str(run))
    except:
        run_spec = False

    for runs in range(5):
        if run_spec:
            runs += run

        for game in games:

            # gym version 0.25.2
            # ie pre 5 arg step
            env = make_env(game, eval=False)

            print(env.observation_space)
            print(env.action_space)

            agent = Agent(n_actions=env.action_space.n, input_dims=[4, 84, 84], total_frames=100000, device=device,
                          game=game, run=runs, name=agent_name)

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
                observation = env.reset()
                while not done and steps < n_steps:
                    steps += 1
                    action = agent.choose_action(observation)
                    observation_, reward, done_, info = env.step(action)

                    time_limit = 'TimeLimit.truncated' in info
                    done = info['game_over'] or time_limit

                    score += reward
                    reward = np.clip(reward, -1., 1.)

                    agent.store_transition(observation, action, reward,
                                           observation_, done_)

                    agent.learn()

                    observation = deepcopy(observation_)

                if steps < n_steps:
                    scores.append([score, steps])
                    scores_temp.append(score)

                    avg_score = np.mean(scores_temp[-50:])

                    if episodes % 1 == 0:
                        print('{} {} avg score {:.2f} total_steps {:.0f} fps {:.2f}'
                              .format(agent_name, game, avg_score, steps, steps / (time.time() - start)), flush=True)

            fname = agent_name + game + "Experiment (" + str(runs) + ').npy'
            np.save(fname, np.array(scores))
            env = make_env(game, eval=True)
            agent.set_eval_mode()
            evals = []
            steps = 0
            eval_episodes = 0
            while eval_episodes < 100:
                done = False
                observation = env.reset()
                score = 0
                while not done:
                    steps += 1
                    action = agent.choose_action(observation)
                    observation_, reward, _, info = env.step(action)

                    time_limit = 'TimeLimit.truncated' in info
                    done = info['game_over'] or time_limit

                    score += reward
                    observation = observation_

                evals.append(score)
                print("Evaluation Score: " + str(score))
                eval_episodes += 1

            fname = agent_name + game + "Evaluation (" + str(runs) + ').npy'
            np.save(fname, np.array(evals))
