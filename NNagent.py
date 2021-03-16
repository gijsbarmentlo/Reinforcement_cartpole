import gym
import numpy as np
import collections
from random import random, randint
import math
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

#Dependencies
import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import initializers
import tensorflow as tf


PLOT_INTEGRATION_CST = 100
BUCKETS = [9, 9, 30, 30]
DISCOUNT_RATE = 0.90
MIN_EPSILON = 0.1
MAX_LR = 0.1
MIN_LR = 0.005
DECAY = 0.7
NUM_ITER = 10000
MAX_TIME = 300
EPOCHS = 3
TRANSMISSION_EPS = 30

class NNCartpoleAgentQ():
    def __init__(self, buckets=BUCKETS, min_lr=MIN_LR, max_lr=MAX_LR, discount_rate=DISCOUNT_RATE,
                 min_epsilon=MIN_EPSILON, decay=DECAY, num_iter=NUM_ITER, epochs=EPOCHS, max_time=MAX_TIME, transmission_eps = TRANSMISSION_EPS):
        self.buckets = buckets
        self.min_lr = min_lr
        self.max_lr = max_lr
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
            self.decay = DECAY

        self.episode_memory = []
        self.epochs = int(epochs)

        # Neural networks
        self.transmission_eps = transmission_eps

        self.m1 = Sequential()
        self.m1.add(Dense(12, input_dim=4, activation="relu"))
        self.m1.add(Dense(24, activation="relu"))
        self.m1.add(Dense(24, activation="relu"))
        self.m1.add(Dense(48, activation="relu"))
        self.m1.add(Dense(2, activation="linear"))
        self.m1.compile(loss='sgd', optimizer='mse', metrics=['mse'])

        self.m2 = tf.keras.models.clone_model(self.m1)

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

    def get_lr(self, e):
        return max(self.min_lr, self.max_lr * (1 - (e / (self.decay * self.num_iter))))

    def choose_action(self, obs, e, learn=False):
        r = random()
        if (r < self.get_epsilon(e)) and learn:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.m2.predict(obs))


    def learn(self, plot = False):
        for episode in tqdm(range(self.num_iter)):
            obs_current = self.env.reset()
            done = False

            while not done:
                self.episode_duration[episode] += 1
                action = self.choose_action(obs_current, episode, learn = True)
                obs_new, reward, done, info = self.env.step(action)
                self.update_nn(reward, obs_current, action, obs_new, episode)
                self.episode_memory.append((reward, obs_current, action, obs_new)) #TODO change data format to deque O(1) complexity instead of O(n)
                obs_current = obs_new

            if (episode%self.transmission_eps)==0:
                self.m2.set_weights(self.m1.get_weights())

            if plot and (episode%1000 ==0):
                self.plot_learning()
                #TODO make interactive plot

        self.memory_replay()

    def update_nn(self, reward, obs_current, action, obs_new, episode):
        target = reward + self.discount_rate * max(self.m2.predict(obs_new))
        if action == 0:
            target = [target, self.m2.predict(obs_current)[1]]
        else:
            target = [self.m2.predict(obs_current)[0], target]

        self.m1.fit(obs_current, target, shuffle = False, )

    def memory_replay(self):
        for i in range(int(self.num_iter * self.epochs)):
            self.updateq(*self.episode_memory[randint(0, self.num_iter-1)], self.num_iter)


    def updateq(self, reward, obs_current, action, obs_new, episode):
        self.qtable[obs_current][action] = (1 - self.get_lr(episode)) * self.qtable[obs_current][action] + \
                                           self.get_lr(episode) * (reward + self.discount_rate * max(self.qtable[obs_new]))


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

        sns.lineplot(x = range(len(episode_duration_avg)), y = episode_duration_avg)
        plt.xlabel("Episode")
        plt.ylabel("Averaged episode duration")
        plt.show()