import time

import keras
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
gamma = 0.99

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r
"""Weighted Gaussian log likelihood loss function"""
def CustomLossGaussian(state, action, reward):
    # Obtain mu and sigma from actor network
    nn_mu, nn_sigma = actor_network(state.reshape(-1, 4))
    # Obtain pdf of Gaussian distribution
    pdf_value = tf.exp(-0.5 * ((action - nn_mu) / (nn_sigma)) ** 2) * 1 / (nn_sigma * tf.sqrt(2 * np.pi))
    # Compute log probability
    log_probability = tf.math.log(pdf_value + 1e-5)
    # Compute weighted loss
    loss_actor = - reward * log_probability
    return loss_actor

class SimpleAgent(keras.Model):
    def __init__(self, s_size, a_size, h_size):
        super(SimpleAgent, self).__init__()
        # Input() can create a placeholder from an arbitrary tf.TypeSpec
        # self.state_in = keras.Input(type_spec=tf.RaggedTensorSpec(shape=[None, s_size], dtype=tf.float32))
        self.state_in = keras.Input(shape=[None, s_size], dtype=tf.float32)
        self.hidden1 = layers.Dense(h_size, activation="relu")
        self.hidden2 = layers.Dense(h_size, activation="relu")

        self.mu = layers.Dense(a_size, activation="linear", kernel_initializer=initializers.Zeros())# , bias_initializer=initializers.Constant(bias_mu)
        self.sigma = layers.Dense(a_size, activation="softplus", kernel_initializer=initializers.Zeros())# , bias_initializer=initializers.Constant(bias_sigma)

    def call(self, inputs, training=False, mask=None):
        x = self.state_in = inputs
        x = self.hidden1(x, training=training)
        x = self.hidden2(x, training=training)
        return [self.mu(x, training=training), self.sigma(x, training=training)]

actor_network = SimpleAgent(s_size=4, a_size=1, h_size=8)
print(actor_network.trainable_variables)
max_angle = 0.418
opt = keras.optimizers.Adam(learning_rate=0.001)

update_frequency = 5
i = 0
total_reward = []
total_length = []
total_episodes = 10000
max_episode = 9999

gradBuffer = None
# gradBuffer = actor_network.trainable_variables
# for ix,grad in enumerate(gradBuffer): gradBuffer[ix] = grad * 0
while i < total_episodes:
    state = env.reset()
    running_reward = 0
    ep_history = []
    for j in range(max_episode):
        mu, sigma = actor_network(state.reshape(-1, 4))         # Obtain mu and sigma from network
        action = tf.random.normal([1], mean=mu, stddev=sigma)   # Draw action from normal distribution
        # | Num | Action                 |
        # |-----|------------------------|
        # | 0   | Push cart to the left  |
        # | 1   | Push cart to the right |
        cart_action = 1 if action.numpy().reshape(-1) > 0 else 0 # threshold action since cart_action is discrete
        next_state, reward, d, _ = env.step(cart_action)
        # if i % 100 == 0 and i!=0: env.render()
        delta_angle =  abs(state[2])    # manually calculate reward from falling angle
        reward = 1 - (delta_angle / max_angle)
        # env.render()
        # print(reward, d)
        # time.sleep(0.1)
        ep_history.append([state, action, reward, next_state])
        running_reward += reward
        if d==True:
            # Update the network
            ep_history = np.array(ep_history)
            ep_history[:, 2] = discount_rewards(ep_history[:, 2])
            if gradBuffer is None:  # init gradBuffer
                gradBuffer = actor_network.trainable_variables
                for ix, grad in enumerate(gradBuffer): gradBuffer[ix] = grad * 0
            with tf.GradientTape() as tape:
                # Compute Gaussian loss
                loss_value = CustomLossGaussian(state, action, reward)
                # Compute gradients
                grads = tape.gradient(loss_value, actor_network.trainable_variables)
                # Apply gradients to update network weights
                # opt.apply_gradients(zip(grads, actor_network.trainable_variables))

            for idx, grad in enumerate(grads): gradBuffer[idx] += grad
            if i % update_frequency == 0 and i != 0:
                # Apply gradients to update network weights
                opt.apply_gradients(zip(gradBuffer, actor_network.trainable_variables))
                for ix, grad in enumerate(gradBuffer): gradBuffer[ix] = grad * 0    # reset buffer
            total_reward.append(running_reward)
            total_length.append(j)
            break
        state = next_state
    if i % 100 == 0: print(np.mean(total_reward[-100:]))
    i += 1