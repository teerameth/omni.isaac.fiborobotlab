from env import MoonCakeEnv
import gym
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

my_env = MoonCakeEnv(headless=False)
# env = gym.make("Pendulum-v1")

# The noise objects for DDPG
n_actions = my_env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))

model = DDPG("MlpPolicy", my_env, verbose=1)
# model = DDPG("MlpPolicy", my_env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)
model.save("ddpg_pendulum")
# env = model.get_env()
#
# del model # remove to demonstrate saving and loading
#
# model = DDPG.load("ddpg_pendulum")
#
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()