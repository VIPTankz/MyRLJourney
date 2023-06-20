import gym
import numpy as np
from gym.wrappers import AtariPreprocessing
import time
from copy import deepcopy

if __name__ == '__main__':

    from DrQ_Agent import Agent
    agent_name = "DrQ"

    games = ["Alien","Amidar","Assault","Asterix","BankHeist","BattleZone","Boxing","Breakout","ChopperCommand","CrazyClimber",\
             "DemonAttack","Freeway","Frostbite","Gopher","Hero","Jamesbond","Kangaroo","Krull","KungFuMaster",\
             "MsPacman","Pong","PrivateEye","Qbert","RoadRunner","Seaquest","UpNDown"]
    for game in games:
        for runs in range(5):
            env = gym.make('ALE/' + game + '-v5')
            env = AtariPreprocessing(env, frame_skip=1)
            env = gym.wrappers.FrameStack(env, 4)
            env.seed(runs)
            print(env.observation_space)
            print(env.action_space)

            agent = Agent(discount=0.99, batch_size=32, n_actions=env.action_space.n,
                           input_dims=[4,84,84], lr=0.0001,
                          max_mem_size=100000)

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
                observation, info = env.reset()
                while not done and not trun:
                    steps += 1
                    action = agent.choose_action(observation)
                    observation_, reward, done, trun, info = env.step(action)
                    score += reward
                    reward = np.clip(reward, -1., 1.)

                    agent.store_transition(observation, action, reward,
                                                  observation_, done)
                    agent.learn()
                    observation = deepcopy(observation_)
                scores.append([score,steps])
                scores_temp.append(score)

                avg_score = np.mean(scores_temp[-50:])

                if episodes % 2 == 0:
                    print('{} episode {} score {:.0f} avg score {:.2f} total_steps {:.0f}'
                          .format(agent_name, episodes, score, avg_score, steps))
                    print("FPS: " + str(time.time() - start))
                    start = time.time()

            fname = agent_name + game + "Experiment (" + str(runs) + ').npy'
            np.save(fname, np.array(scores))
            agent.eval_mode = True
            evals = []
            steps = 0
            while steps < 125000:
                done = False
                trun = False
                observation, info = env.reset()
                score = 0
                while not done and not trun:
                    steps += 1
                    action = agent.choose_action(observation)
                    observation_, reward, done, trun, info = env.step(action)
                    score += reward
                    observation = observation_

                evals.append(score)
                print("Evaluation Score: " + str(score))

            fname = agent_name + game + "Evaluation (" + str(runs) + ').npy'
            np.save(fname, np.array(evals))