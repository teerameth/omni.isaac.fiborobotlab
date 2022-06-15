import numpy as np
import gym

import wandb
from wandb.integration.sb3 import WandbCallback

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

config = {
    "policy_type": "MlpLstmPolicy",
    "total_timesteps": 25000,
    "env_name": "CartPole-v1",
}

run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
)


env = gym.make('CartPole-v1')
model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)
run.finish()
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, warn=False)
# print(mean_reward)

model.save("ppo_recurrent")
del model # remove to demonstrate saving and loading

model = RecurrentPPO.load("ppo_recurrent")

obs = env.reset()
# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
while True:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    episode_starts = dones
    env.render()
    if dones:
        lstm_states = None
        env.reset()