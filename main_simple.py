import torch
import gym
import time
from copy import deepcopy
from M_DDQN import Agent
import numpy as np
def make_env(game, eval):
    env = gym.make(game)

    return env

if __name__ == "__main__":
    agent_name = "DDQN"
    total_frames = 20000

    games = ["CartPole-v1","LunarLander-v2","Acrobot-v1","MountainCar-v0"]

    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")

    for runs in range(1):
        for game in games:

            env = make_env(game, eval=False)

            print(env.observation_space)
            print(env.action_space)

            agent = Agent(n_actions=env.action_space.n, input_dims=env.observation_space.shape, total_frames=total_frames, device=device,
                          game=game, run=runs, name=agent_name)

            scores = []
            scores_temp = []
            n_steps = total_frames
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
                    observation_, reward, done, info = env.step(action)
                    #env.render()

                    score += reward

                    agent.store_transition(observation, action, reward,
                                           observation_, done)

                    agent.learn()

                    observation = deepcopy(observation_)

                if steps < n_steps:
                    scores.append([score, steps])
                    scores_temp.append(score)

                    avg_score = np.mean(scores_temp[-10:])

                    if episodes % 1 == 0:
                        print('{} {} avg score {:.2f} total_steps {:.0f} fps {:.2f}'
                              .format(agent_name, game, avg_score, steps, steps / (time.time() - start)), flush=True)

            fname = agent_name + game + "Experiment (" + str(runs) + ').npy'
            np.save(fname, np.array(scores))