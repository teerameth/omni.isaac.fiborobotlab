import collections
import time
import sys

import gym
import numpy as np
import statistics
import tensorflow as tf
import tqdm
import math

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple

import carb
from omni.isaac.kit import SimulationApp
from omni.isaac.imu_sensor import _imu_sensor

## Create environments ##
from env_obike import Env
env = Env(physics_dt=1/100, rendering_dt=1/30, headless=False)

state = env.reset()

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(
        self,
        num_actions: int,
        num_hidden_units: int):
        """Initialize."""
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation="relu")
        # self.lstm = layers.LSTM(units=num_hidden_units, activation="relu", recurrent_activation="sigmoid", stateful=True)
        self.actor = layers.Dense(num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor, states = None) -> Tuple[tf.Tensor, tf.Tensor]:
        x = inputs
        x = self.common(x)
        # if states is None: states = self.lstm.get_initial_state(x)
        # else: states = self.lstm.states
        # x = self.lstm(inputs, initial_state=states)
        return self.actor(x), self.critic(x)

num_actions = env.num_action_space # 1
num_hidden_units = 16

model = ActorCritic(num_actions, num_hidden_units)

# Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns state, reward and done flag given an action."""
    state, reward, done, _ = env.step(action, render=True)
    return (state.astype(np.float32),
            np.array(reward, np.int32),
            np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(env_step, [action], [tf.float32, tf.int32, tf.int32])

def run_episode(
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Runs a single episode to collect training data."""

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        state = tf.expand_dims(state, 0)

        # Run the model and to get action mean and S.D.
        action_logits_t, value = model(state)

        # Sample next action from the action probability distribution
        # action = tf.random.categorical(action_logits_t, 1)[0, 0]
        action = tf.random.normal(shape=[1], mean=action_logits_t, stddev=value)[0][0]
        action_probs_t = tf.nn.softmax(action_logits_t)

        # Store critic values
        values = values.write(t, tf.squeeze(value))

        # Store log probability of the action chosen
        # action_probs = action_probs.write(t, action_probs_t[0, action])
        action_probs = action_probs.write(t, action_logits_t[0, 0])

        # Apply action to the environment to get next state and reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()

    return action_probs, values, rewards

def get_expected_return(
        rewards: tf.Tensor,
        gamma: float,
        standardize: bool = True) -> tf.Tensor:
    """Compute expected returns per timestep."""

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps))
    return returns

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(
      action_probs: tf.Tensor,
      values: tf.Tensor,
      returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined actor-critic loss."""

    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# @tf.function
def train_step(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    gamma: float,
    max_steps_per_episode: int) -> tf.Tensor:
    """Runs a model training step."""
    with tf.GradientTape() as tape:
        # Run the model for one episode to collect training data
        action_probs, values, rewards = run_episode(initial_state, model, max_steps_per_episode)

        # Calculate expected returns
        returns = get_expected_return(rewards, gamma)
        # Convert training data to appropriate TF tensor shapes
        action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]
        # Calculating loss values to update our network
        loss = compute_loss(action_probs, values, returns)
    # Compute the gradients from the loss
    grads = tape.gradient(loss, model.trainable_variables)
    # Apply the gradients to the model's parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    episode_reward = tf.math.reduce_sum(rewards)
    return episode_reward


min_episodes_criterion = 100
max_episodes = 10000
max_steps_per_episode = 1000

## O-bike balancing is considered solved if average reward is >= 180 over 100 consecutive trials
reward_threshold = 180
running_reward = 0

# Discount factor for future rewards
gamma = 0.99

# Keep last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

with tqdm.trange(max_episodes) as ep:
    for i in ep:
        initial_state = tf.constant(env.reset(), dtype=tf.float32)
        with tf.GradientTape() as tape:
            # Run the model for one episode to collect training data
            action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

            initial_state_shape = initial_state.shape
            state = initial_state

            for t in tf.range(max_steps_per_episode):
                # Convert state into a batched tensor (batch size = 1)
                state = tf.expand_dims(state, 0)
                # print("STATE")
                # print(state)
                # Run the model and to get action mean and S.D.
                action_logits_t, value = model(state)
                # print("action_logits_t, VALUE")
                # print(action_logits_t, value)
                # Sample next action from the action probability distribution
                # action = tf.random.categorical(action_logits_t, 1)[0, 0]
                action = tf.random.normal(shape=[1], mean=action_logits_t, stddev=value)[0][0]
                action_probs_t = tf.nn.softmax(action_logits_t)

                # Store critic values
                values = values.write(t, tf.squeeze(value))

                # Store log probability of the action chosen
                # action_probs = action_probs.write(t, action_probs_t[0, action])
                action_probs = action_probs.write(t, action_logits_t[0, 0])

                # Apply action to the environment to get next state and reward
                # print("ACTION")
                # print(action)
                state, reward, done, _ = env.step(action)
                # state, reward, done = tf_env_step(action)
                # state.set_shape(initial_state_shape)

                # Store reward
                rewards = rewards.write(t, reward)
                if tf.cast(done, tf.bool):
                    break
            action_probs = action_probs.stack()
            values = values.stack()
            rewards = rewards.stack()
            # action_probs, values, rewards = run_episode(initial_state, model, max_steps_per_episode)
            # Calculate expected returns
            returns = get_expected_return(rewards, gamma)
            # Convert training data to appropriate TF tensor shapes
            action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]]
            # Calculating loss values to update our network
            loss = compute_loss(action_probs, values, returns)
        # Compute the gradients from the loss
        grads = tape.gradient(loss, model.trainable_variables)
        # Apply the gradients to the model's parameters
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        episode_reward = int(tf.math.reduce_sum(rewards))
        # episode_reward = int(train_step(initial_state, model, optimizer, gamma, max_steps_per_episode))

        episodes_reward.append(episode_reward)
        running_reward = statistics.mean(episodes_reward)

        ep.set_description(f'Episode {i}')
        ep.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

        # Show average episode reward every 10 episodes
        if i % 10 == 0:
            pass  # print(f'Episode {i}: average reward: {avg_reward}')

        if running_reward > reward_threshold and i >= min_episodes_criterion:
            break

print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')