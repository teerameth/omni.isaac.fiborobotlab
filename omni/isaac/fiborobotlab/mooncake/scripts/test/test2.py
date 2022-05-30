import ray
from ray.rllib.agents.ppo import PPOTrainer
ray.init() # Skip or set to ignore if already called
config = {'gamma': 0.9,
          'lr': 1e-2,
          'num_workers': 4,
          'train_batch_size': 1000,
          'model': {
              'fcnet_hiddens': [128, 128]
          }}
trainer = PPOTrainer(env='CartPole-v0', config=config)
for i in range(100):
    print(trainer.train())