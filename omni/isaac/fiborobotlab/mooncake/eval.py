from env2 import MoonCakeEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback

policy_path = "./mlp_policy/mooncake_policy_checkpoint_100000_steps.zip.zip"

my_env = MoonCakeEnv(headless=False)
model = PPO.load(policy_path)

for _ in range(20):
    obs = my_env.reset()
    done = False
    while not done:
        actions, _ = model.predict(observation=obs, deterministic=True)
        obs, reward, done, info = my_env.step(actions)
        my_env.render()

my_env.close()