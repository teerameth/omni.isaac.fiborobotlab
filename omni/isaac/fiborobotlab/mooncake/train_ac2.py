from env import MoonCakeEnv
import gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

my_env = MoonCakeEnv(headless=False)
# Parallel environments
# env = make_vec_env("CartPole-v1", n_envs=4)

model = A2C("MlpPolicy", my_env, verbose=1)
model.learn(total_timesteps=5000000)
model.save("a2c_cartpole")

# del model # remove to demonstrate saving and loading
#
# model = A2C.load("a2c_cartpole")
#
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()