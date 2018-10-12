from collections import deque
import time
import os
import numpy as np
import torch

from dqn_agent import Agent


class Runner():

    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    def __init__(self, env):
        self.env = env

        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

    def run(self, run_id = 1, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, lr=5e-4, use_double_dqn=False, use_soft_update=True):
        start = time.time()

        agent = Agent(state_size=37, action_size=4, seed=0, lr=lr, use_double_dqn=use_double_dqn, use_soft_update=use_soft_update)

        # list containing scores from each episode
        scores = []

        # last 100 scores
        scores_window = deque(maxlen=100)

        # initialize epsilon
        eps = eps_start

        for i_episode in range(1, n_episodes + 1):

            # reset the environment
            env_info = self.env.reset(train_mode=True)[self.brain_name]

            # get the current state
            state = env_info.vector_observations[0]

            score = 0

            for t in range(max_t):

                action = agent.act(state, eps)
                #print("action: ", action)

                # send the action to the environment
                env_info = self.env.step(action)[self.brain_name]

                # get the next state
                next_state = env_info.vector_observations[0]

                # get the reward
                reward = env_info.rewards[0]

                # see if episode has finished
                done = env_info.local_done[0]

                # TODO add proper comment
                agent.step(state, action, reward, next_state, done)

                state = next_state
                score += reward

                if done:
                    break

            # save most recent score
            scores_window.append(score)

            # save most recent score
            scores.append(score)

            # decrease epsilon
            eps = max(eps_end, eps_decay * eps)

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

            if np.mean(scores_window)>=14.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))

                end = time.time()
                elapsed = end - start
                print("\nTime taken to solve: {:.2f} minutes".format(elapsed / 60.0))

                run_dir = "results/{}".format(run_id)
                os.mkdir(run_dir)

                torch.save(agent.qnetwork_local.state_dict(), "{}/checkpoint.pth".format(run_dir))
                break

        return scores




