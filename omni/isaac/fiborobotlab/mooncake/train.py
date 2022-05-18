from env import MoonCakeEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th

log_dir = "./mlp_policy"
# set headles to false to visualize training
my_env = MoonCakeEnv(headless=False)

policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[16, dict(pi=[64, 32], vf=[64, 32])])
policy = CnnPolicy
total_timesteps = 500000

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="mooncake_policy_checkpoint")
model = PPO(
    policy,
    my_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=10000,
    batch_size=1000,
    learning_rate=0.00025,
    gamma=0.9995,
    device="cuda",
    ent_coef=0,
    vf_coef=0.5,
    max_grad_norm=10,
    tensorboard_log=log_dir,
)
model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])

model.save(log_dir + "/mooncake_policy")

my_env.close()
