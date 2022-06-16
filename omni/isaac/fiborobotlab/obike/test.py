import time
import tensorflow as tf
import random
from env_obike import ObikeEnv
env = ObikeEnv(headless=False)
state = env.reset()
print(state)
with tf.GradientTape() as tape:
    while True:
        state, reward, done, _ = env.step(action=0.5)
        print(state, reward, done)
        if done:
            state = env.reset()
            print(state)