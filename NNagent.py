import gym
import numpy as np
from collections import deque
from random import random, randint, sample
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
from keras.optimizers import Adam



PLOT_INTEGRATION_CST = 10
DISCOUNT_RATE = 0.90
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995
ALPHA = 0.01
ALPHA_DECAY = 0.01
DECAY = 0.7
MAX_TIME = 300
EPOCHS = 2
TRANSMISSION_EPS = 6 #N_update
BATCH_SIZE = 128

class CartpoleAgentNN():
    def __init__(self, alpha=ALPHA, alpha_decay=ALPHA_DECAY, discount_rate=DISCOUNT_RATE,
                 min_epsilon=MIN_EPSILON, epsilon_decay=EPSILON_DECAY, decay=DECAY, epochs=EPOCHS, max_time=MAX_TIME,
                 transmission_eps=TRANSMISSION_EPS, batch_size=BATCH_SIZE):

        self.discount_rate = discount_rate
        self.epsilon = 0.9
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.env = gym.make('CartPole-v0')
        self.max_time = max_time
        self.episode_duration = []
        if 1 >= decay >= 0:
            self.decay = decay
        else:
            self.decay = DECAY

        self.memory = deque()
        self.epochs = epochs

        # Neural networks
        self.transmission_eps = transmission_eps
        self.batch_size = batch_size
        self.training_batch_x = np.zeros((batch_size, 4))
        self.training_batch_y = np.zeros((batch_size, 2))

        self.learning_model = Sequential()
        self.learning_model.add(Dense(24, input_dim=4, activation='tanh'))
        self.learning_model.add(Dense(48, activation='tanh'))
        self.learning_model.add(Dense(2, activation='linear'))
        self.learning_model.compile(loss='mse', optimizer=Adam(lr=alpha, decay=alpha_decay))

        self.prediction_model = tf.keras.models.clone_model(self.learning_model)
        self.prediction_model.compile(loss='mse', optimizer=Adam(lr=alpha, decay=alpha_decay))

    def normalise(self, obs): #TODO implement normalise and use kill angle as max angle

        max_obs = [2.4, 3.0, 12 * 2 * math.pi / 360, 3.0]
        return obs/max_obs

    def choose_action(self, obs, e, learn=False):
        return self.env.action_space.sample() if (random() < self.epsilon) and learn else np.argmax(self.prediction_model.predict(obs))

    def memory_learn(self, t):
        # x_batch, y_batch = [], []
        # minibatch = sample(self.memory, min(len(self.memory), self.batch_size))
        # for reward, obs_current, action, obs_new, done in minibatch:
        #     y = self.prediction_model.predict(obs_current)[0]
        #     y[action] = reward + max(self.prediction_model.predict(obs_new)[0]) * self.discount_rate if done else reward
        #     x_batch.append(obs_current[0])
        #     y_batch.append(y)

        # temp_memory = self.memory.copy()
        # x_batch, y_batch = [], []
        # for i in range(min(len(self.memory), self.batch_size)):
        #     reward, obs_current, action, obs_new, done = temp_memory.pop()
        #     y = self.prediction_model.predict(obs_current)[0]
        #     y[action] = reward + max(self.prediction_model.predict(obs_new)[0]) * self.discount_rate if done else reward
        #     x_batch.append(obs_current[0])
        #     y_batch.append(y)
        #
        # self.learning_model.fit(np.reshape(x_batch, (len(x_batch), 4)), np.reshape(y_batch, (len(x_batch), 2)), batch_size=len(x_batch), verbose=0)
        # self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        x_batch, y_batch = [], []
        for i in range(max(t, self.batch_size)):
            reward, obs_current, action, obs_new, done = self.memory.pop()
            self.memory.appendleft((reward, obs_current, action, obs_new, done))
            y = self.prediction_model.predict(obs_current)[0]
            y[action] = reward + max(self.prediction_model.predict(obs_new)[0]) * self.discount_rate if done else reward
            x_batch.append(obs_current[0])
            y_batch.append(y)

        self.learning_model.fit(np.reshape(x_batch, (len(x_batch), 4)), np.reshape(y_batch, (len(x_batch), 2)),
                                batch_size=self.batch_size, verbose=0)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


    def run(self, num_episodes, plot=False):
        for episode in tqdm(range(num_episodes)):
            obs_current = np.reshape(self.normalise(self.env.reset()), [1, 4])
            done = False
            t = 0
            while not done:
                t += 1
                action = self.choose_action(obs_current, episode, learn=True)
                obs_new, reward, done, info = self.env.step(action)
                obs_new = np.reshape(self.normalise(obs_new), [1, 4])
                reward = self.optimise_reward(reward, obs_new)
                self.memory.append((reward, obs_current, action, obs_new, done))
                obs_current = obs_new

            self.episode_duration.append(t)
            self.memory_learn(t)

            if (episode % self.transmission_eps) == self.transmission_eps:
                self.prediction_model.set_weights(self.learning_model.get_weights())


    def optimise_reward(self, reward, obs):
        #punish if loss
        reward = -10 if reward == 0 else reward
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # self.x_threshold = 2.4
        if abs(obs[0, 3]) < 8 * 2 * math.pi / 360:
            reward += 10
            if abs(obs[0 ,3]) < 4 * 2 * math.pi / 360:
                reward += 10

        if abs(obs[0, 0]) < 1.6:
            reward += 10
            if abs(obs[0, 0]) < 0.8:
                reward += 10
        return reward

    def memory_replay(self):
        print("starting memory replay")
        training_batch_x = np.zeros((self.batch_size, 4))
        training_batch_y = np.zeros((self.batch_size, 2))

        for epoch in range(self.epochs):
            temp_memory = self.memory.copy()
            x_batch, y_batch = [], []

            for i in range(len(self.memory)):
                #Generate training batch from memory
                reward, obs_current, action, obs_new, done = temp_memory.pop()
                y = self.prediction_model.predict(obs_current)[0]
                y[action] = reward + max(self.prediction_model.predict(obs_new)[0]) * self.discount_rate if done else reward
                x_batch.append(obs_current[0])
                y_batch.append(y)

                #Train model when x_batch has reached batch_size
                if i % self.batch_size == self.batch_size:
                    self.learning_model.fit(np.reshape(x_batch, (len(x_batch), 4)),
                                            np.reshape(y_batch, (len(x_batch), 2)), batch_size=self.batch_size,
                                            verbose=0)
                    x_batch, y_batch = [], []

                #Update prediction model after transmission_eps batches
                if int(i/self.batch_size) == self.transmission_eps:
                    self.prediction_model.set_weights(self.learning_model.get_weights())

            #train onr remainder of x_batch and y_batch upon exiting the loop and make prediction model catch up
            self.learning_model.fit(np.reshape(x_batch, (len(x_batch), 4)), np.reshape(y_batch, (len(x_batch), 2)), batch_size=self.batch_size, verbose=0)
            self.prediction_model.set_weights(self.learning_model.get_weights())


    def show(self, episodes=10):
        for episode in range(episodes):
            obs = np.reshape(self.env.reset(), [1, 4])
            for t in range(self.max_time):
                self.env.render()
                action = self.choose_action(obs, episode, learn=False)
                obs, reward, done, info = self.env.step(action)
                obs = np.reshape(obs, [1, 4])
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break

    def plot_learning(self):
        """
        Plots the number of steps at each episode and prints the
        amount of times that an episode was successfully completed.
        """
        episode_duration_avg = []
        for i in range(math.floor(len(self.episode_duration)/PLOT_INTEGRATION_CST)):
            episode_duration_avg.append(sum(self.episode_duration[PLOT_INTEGRATION_CST*i:PLOT_INTEGRATION_CST*(i+1)])/PLOT_INTEGRATION_CST)

        sns.lineplot(x=range(len(episode_duration_avg)), y=episode_duration_avg)
        plt.xlabel("Episode")
        plt.ylabel("Averaged episode duration")
        plt.show()