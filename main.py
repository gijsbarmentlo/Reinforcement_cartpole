# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import gym
import numpy as np
from random import random, choice
import math
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Config
BUCKETS = [5, 7, 20, 10]
LEARNING_RATE = 0.01
DISCOUNT_RATE = 0.90
EPSILON = 0.9
NUM_ITER = 300
MAX_TIME = 220
PLOT_INTEGRATION_CST = 100


def test_discretise(agent):
    fake_obs = agent.env.reset()
    discretised_obs = agent.discretise(fake_obs)
    print(f"fake obs = {fake_obs}, discretised = {discretised_obs}")

    fake_obs = [-3.4, 0.1, -0.2, 5.7]
    discretised_obs = agent.discretise(fake_obs)
    print(f"fake obs = {fake_obs}, discretised = {discretised_obs}")

    fake_obs = [4.8, 10, -0.418, 10]
    discretised_obs = agent.discretise(fake_obs)
    print(f"fake obs = {fake_obs}, discretised = {discretised_obs}")


def test_agent(agent):
    test_discretise(agent)


class CartpoleAgent():
    def __init__(self, buckets, learning_rate, discount_rate, epsilon, num_iter, max_time):
        self.buckets = buckets
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.env = gym.make('CartPole-v0')
        self.qtable = np.zeros(tuple(buckets) + (2,))
        self.num_iter = num_iter
        self.max_time = max_time
        self.episode_duration = np.zeros(num_iter)


    def discretise(self, obs):

        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2],
                        math.radians(50) / 1.]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2],
                        -math.radians(50) / 1.]

        discretized = list()
        for i in range(len(obs)):
            scaling = ((obs[i] + abs(lower_bounds[i]))
                       / (upper_bounds[i] - lower_bounds[i]))
            new_obs = int(round((self.buckets[i] - 1) * scaling))
            new_obs = min(self.buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)


    def choose_action(self, obs, learn=False):
        r = random()
        if (r < self.epsilon) and learn:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.qtable[obs])


    def learn(self, plot = False):
        for episode in tqdm(range(self.num_iter)):
            obs_current = self.discretise(self.env.reset())

            for t in range(self.max_time):
                self.episode_duration[episode] += 1
                action = self.choose_action(obs_current, learn = True)
                obs_new, reward, done, info = self.env.step(action)
                obs_new = self.discretise(obs_new)
                self.updateq(reward, obs_current, action, obs_new)
                obs_current = obs_new
                if done:
                    break

            if plot and (episode%1000 ==0):
                self.plot_learning()
                #TODO make interactive plot


    def updateq(self, reward, obs_current, action, obs_new):
        self.qtable[obs_current][action] = (1 - self.learning_rate) * self.qtable[obs_current][action] + self.learning_rate * (reward + self.discount_rate * max(self.qtable[obs_new]))


    def show(self, episodes=10):
        for i_episode in range(episodes):
            obs = self.discretise(self.env.reset())
            for t in range(self.max_time):
                self.env.render()
                action = self.choose_action(obs, learn=False)
                obs, reward, done, info = self.env.step(action)
                obs = self.discretise(obs)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break

    def plot_learning(self):
        """
        Plots the number of steps at each episode and prints the
        amount of times that an episode was successfully completed.
        """
        episode_duration_avg = []
        for i in range(math.floor(self.num_iter/PLOT_INTEGRATION_CST)):
            episode_duration_avg.append(sum(self.episode_duration[PLOT_INTEGRATION_CST*i : PLOT_INTEGRATION_CST*(i+1)])/PLOT_INTEGRATION_CST)

        sns.lineplot(x = range(len(self.episode_duration)), y = self.episode_duration)
        plt.xlabel("Episode")
        plt.ylabel("Averaged episode duration")
        plt.show()


if __name__ == '__main__':
    agent = CartpoleAgent(BUCKETS, LEARNING_RATE, DISCOUNT_RATE, EPSILON, num_iter=100000, max_time=MAX_TIME)
    agent.learn()
    agent.show(10)
    agent.plot_learning()

