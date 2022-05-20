from env import MoonCakeEnv
import gym

from stable_baselines3 import DQN
my_env = MoonCakeEnv(headless=False)
# env = gym.make("CartPole-v0")

model = DQN("MlpPolicy", my_env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_cartpole")

# del model # remove to demonstrate saving and loading
#
# model = DQN.load("dqn_cartpole")
#
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()