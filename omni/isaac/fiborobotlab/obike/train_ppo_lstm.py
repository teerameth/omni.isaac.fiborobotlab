import numpy as np
from env_obike import ObikeEnv
import gym

import numpy as np
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


env = ObikeEnv(skip_frame=1,
               physics_dt=1.0 / 100.0,
               rendering_dt=1.0 / 60.0,
               max_episode_length=60,
               headless=False,)
model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}", device="cpu")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=1000,
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

observations = env.reset()
# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
while True:
    # obs = [observations["lin_acc_y"], observations["lin_acc_z"], observations["ang_vel_x"]]
    # obs = np.array(obs, dtype=np.float32)
    action, lstm_states = model.predict(observations, state=lstm_states, episode_start=episode_starts, deterministic=True)
    observations, rewards, dones, info = env.step(action)
    episode_starts = dones
    env.render()
    if dones:
        lstm_states = None  # Clear internal states
        observations = env.reset()