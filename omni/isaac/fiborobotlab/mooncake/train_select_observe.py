from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
from env_select_observe import MoonCakeEnv
import os

log_dir = "./mlp_policy"
# set headles to false to visualize training

b = [1, 1, 1, 1, 1]
save_dir = log_dir + "/mooncake_policy_" + str(b[0])+str(b[1])+str(b[2])+str(b[3])+str(b[4])
os.mkdir(save_dir)
with open(save_dir + '/log.txt', 'w') as f:
    f.write(str(b) + '\n')
[print("####################################################################################################") for i in range(3)]
print(b)
[print("####################################################################################################") for i in range(3)]
my_env = MoonCakeEnv(headless=True, observ_selection = [b[0], b[1], b[2], b[3], b[4]])   #

# checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="mooncake_policy_checkpoint")
model = PPO(MlpPolicy,
            my_env,
            verbose=1,
            n_steps=10000,
            batch_size=100,
            learning_rate=0.00025,
            )
# model.learn(total_timesteps=500000, callback=[checkpoint_callback])
model.learn(total_timesteps=500000)

model.save(save_dir)
my_env.close()
