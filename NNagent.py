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
from keras.layers import Dense, Dropout
from tensorflow.keras import initializers
import tensorflow as tf



PLOT_INTEGRATION_CST = 100
DISCOUNT_RATE = 0.90
MIN_EPSILON = 0.7
MAX_LR = 0.05
ALPHA = 0.01
ALPHA_DECAY = 0.01
DECAY = 0.7
NUM_ITER = 1000
MAX_TIME = 300
EPOCHS = 2
TRANSMISSION_EPS = 6 #N_update
BATCH_SIZE = 128

class CartpoleAgentNN():
    def __init__(self, min_lr=MIN_LR, max_lr=MAX_LR, discount_rate=DISCOUNT_RATE,
                 min_epsilon=MIN_EPSILON, decay=DECAY, num_iter=NUM_ITER, epochs=EPOCHS, max_time=MAX_TIME,
                 transmission_eps=TRANSMISSION_EPS, batch_size=BATCH_SIZE):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.discount_rate = discount_rate
        self.min_epsilon = min_epsilon
        self.env = gym.make('CartPole-v0')
        self.num_iter = num_iter
        self.max_time = max_time
        self.episode_duration = np.zeros(num_iter)
        if 1 >= decay >= 0:
            self.decay = decay
        else:
            self.decay = DECAY

        self.memory = []
        self.epochs = int(epochs)

        # Neural networks
        self.transmission_eps = transmission_eps
        self.batch_size = batch_size
        self.training_batch_x = np.zeros((batch_size, 4))
        self.training_batch_y = np.zeros((batch_size, 2))

        self.learning_model = Sequential()
        self.learning_model.add(Dense(8, input_dim=4, activation="relu"))
        self.learning_model.add(Dropout(0.2))
        self.learning_model.add(Dense(8, activation="relu"))
        self.learning_model.add(Dropout(0.2))
        self.learning_model.add(Dense(6, activation="relu"))
        self.learning_model.add(Dropout(0.2))
        self.learning_model.add(Dense(4, activation="relu"))
        self.learning_model.add(Dropout(0.2))
        self.learning_model.add(Dense(2, activation="linear"))
        self.learning_model.compile(loss='mse', optimizer ="adam")

        # self.prediction_model = Sequential()
        # self.prediction_model.add(Dense(8, input_dim=4, activation="relu"))
        # self.prediction_model.add(Dropout(0.2))
        # self.prediction_model.add(Dense(8, activation="relu"))
        # self.prediction_model.add(Dropout(0.2))
        # self.prediction_model.add(Dense(6, activation="relu"))
        # self.prediction_model.add(Dropout(0.2))
        # self.prediction_model.add(Dense(4, activation="relu"))
        # self.prediction_model.add(Dropout(0.2))
        # self.prediction_model.add(Dense(2, activation="linear"))
        # self.prediction_model.compile(loss='mse', optimizer="adam")

        self.prediction_model = tf.keras.models.clone_model(self.learning_model)

    def normalise(self, obs): #TODO implement normalise and use kill angle as max angle
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

    def get_epsilon(self, episode):
        return max(self.min_epsilon, 1 - (episode / (self.decay * self.num_iter)))

    def choose_action(self, obs, e, learn=False):
        r = random()
        if (r < self.get_epsilon(e)) and learn:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.prediction_model.predict(obs)[0])

    def learn(self, plot=False):
        for episode in tqdm(range(self.num_iter)):
            obs_current = np.reshape(self.env.reset(), (1, 4))
            done = False
            t = 0

            while not done:
                self.episode_duration[episode] += 1
                t += 1

                action = self.choose_action(obs_current, episode, learn=True)
                obs_new, reward, done, info = self.env.step(action)
                obs_new = np.reshape(obs_new, (1, 4))
                self.update_nn(reward, obs_current, action, obs_new, t)
                self.memory.append((reward, obs_current, action, obs_new)) #TODO change data format to deque O(1) complexity instead of O(n)
                obs_current = obs_new

            if t % self.batch_size != 0: #TODO change condition
                self.learning_model.fit(self.training_batch_x[0:t % self.batch_size], self.training_batch_y[0:t % self.batch_size], shuffle=False, verbose=0)

            if (episode % self.transmission_eps) == 0:
                self.prediction_model.set_weights(self.learning_model.get_weights())

            if plot and (episode%1000 == 0):
                self.plot_learning()
                #TODO make interactive plot

    def update_nn(self, reward, obs_current, action, obs_new, t):
        reward = self.optimise_reward(reward, obs_current)

        y = self.prediction_model.predict(obs_current)[0]
        y[action] = reward + max(self.prediction_model.predict(obs_new)[0]) * self.discount_rate

        self.training_batch_x[t % self.batch_size] = obs_current
        self.training_batch_y[t % self.batch_size] = y #np.reshape(y, (1, 2))

        if t % self.batch_size == 0: #self.batch_size - 1: #TODO read through indices and check
            self.learning_model.fit(self.training_batch_x, self.training_batch_y, shuffle=False, verbose=0)

    def optimise_reward(self, reward, obs):
        #punish if loss
        reward = -10 if reward == 0 else reward
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # self.x_threshold = 2.4
        if   abs(obs[0, 3]) < 8 * 2 * math.pi / 360:
            reward += 10
            if abs(obs[0 ,3]) < 4 * 2 * math.pi / 360:
                reward += 10

        if abs(obs[0, 0]) < 1.6:
            reward += 10
            if abs(obs[0, 0]) < 0.8:
                reward += 10

        return reward

    def memory_replay(self):
        t = 0
        for i in tqdm(range(self.num_iter * self.epochs)):
            t += 1
            self.update_nn(*self.memory[randint(0, len(self.memory) - 1)], t)
            #TODO add backprop for leftover %batchsize


    def show(self, episodes=10):
        for episode in range(episodes):
            obs = np.reshape(self.env.reset(), (1, 4))
            for t in range(self.max_time):
                self.env.render()
                action = self.choose_action(obs, episode, learn=False)
                obs, reward, done, info = self.env.step(action)
                obs = np.reshape(obs, (1, 4))
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