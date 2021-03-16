import gym
import numpy as np
from random import random, randint
import math
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

PLOT_INTEGRATION_CST = 100
DEFAULT_DECAY = 0.7
BUCKETS = [5, 7, 20, 10]
LEARNING_RATE = 0.01
DISCOUNT_RATE = 0.90
MIN_EPSILON = 0.1
DECAY = 0.8
NUM_ITER = 300
MAX_TIME = 300
REPLAY_RATE = 2

class CartpoleAgentQ():
    def __init__(self, buckets=BUCKETS, learning_rate=LEARNING_RATE, discount_rate=DISCOUNT_RATE,
                 min_epsilon=MIN_EPSILON, decay=DECAY, num_iter=NUM_ITER, replay_rate=REPLAY_RATE, max_time=MAX_TIME):
        self.buckets = buckets
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.min_epsilon = min_epsilon
        self.env = gym.make('CartPole-v0')
        self.qtable = np.zeros(tuple(buckets) + (2,))
        self.num_iter = num_iter
        self.max_time = max_time
        self.episode_duration = np.zeros(num_iter)
        if decay <= 1 and 0 <= decay:
            self.decay = decay
        else:
            self.decay = DEFAULT_DECAY

        #self.episode_memory = np.zeros((num_iter, 1))
        self.episode_memory = []
        self.replay_rate = replay_rate

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

    def get_epsilon(self, e):
        return max(self.min_epsilon, 1 - (e / (self.decay * self.num_iter)))


    def choose_action(self, obs, e, learn=False):
        r = random()
        if (r < self.get_epsilon(e)) and learn:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.qtable[obs])


    def learn(self, plot = False):
        for episode in tqdm(range(self.num_iter)):
            obs_current = self.discretise(self.env.reset())

            for t in range(self.max_time):
                self.episode_duration[episode] += 1
                action = self.choose_action(obs_current, episode, learn = True)
                obs_new, reward, done, info = self.env.step(action)
                obs_new = self.discretise(obs_new)
                self.updateq(reward, obs_current, action, obs_new)
                self.episode_memory.append((reward, obs_current, action, obs_new))
                obs_current = obs_new
                if done:
                    break

            if plot and (episode%1000 ==0):
                self.plot_learning()
                #TODO make interactive plot

        self.memory_replay()



    def memory_replay(self):
        for i in range(int(self.num_iter * self.replay_rate)):
            self.updateq(*self.episode_memory[randint(0, self.num_iter-1)])

    def updateq(self, reward, obs_current, action, obs_new):
        self.qtable[obs_current][action] = (1 - self.learning_rate) * self.qtable[obs_current][action] + self.learning_rate * (reward + self.discount_rate * max(self.qtable[obs_new]))


    def show(self, episodes=10):
        for i_episode in range(episodes):
            obs = self.discretise(self.env.reset())
            for t in range(self.max_time):
                self.env.render()
                action = self.choose_action(obs, i_episode, learn=False)
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