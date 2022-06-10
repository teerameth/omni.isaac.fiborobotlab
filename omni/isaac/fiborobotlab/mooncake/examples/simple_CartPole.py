import keras
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.python.framework import ops
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
gamma = 0.99

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class SimpleAgent(keras.Model):
    def __init__(self, s_size, a_size, h_size):
        super(SimpleAgent, self).__init__()
        # Input() can create a placeholder from an arbitrary tf.TypeSpec
        # self.state_in = keras.Input(type_spec=tf.RaggedTensorSpec(shape=[None, s_size], dtype=tf.float32))
        # self.state_in = keras.Input(shape=[None, s_size], dtype=tf.float32)
        self.hidden1 = layers.Dense(h_size)
        self.hidden2 = layers.Dense(a_size, activation='softmax')

        self.chosen_action = None
        # The next six lines establish the training procedure. We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        self.reward_holder = keras.Input(shape=[None], dtype=tf.float32)
        self.action_holder = keras.Input(shape=[None], dtype=tf.int32)

        # self.indexes = tf.range(0, tf.shape(self.hidden2)[0]) * tf.shape(self.hidden2)[1] + self.action_holder
        # self.responsible_outputs = tf.gather(tf.reshape(self.hidden2, [-1]), self.indexes)
        # self.loss = -tf.reduce_mean(tf.math.log(self.responsible_outputs) * self.reward_holder)
        # tvars = self.trainable_variables()
        # self.gradient_holders = []
        # for idx,var in enumerate(tvars):
        #     placeholder = keras.Input(tf.float32,name=str(idx)+'_holder')
        #     self.gradient_holders.append(placeholder)
        # self.gradients = tf.gradients(self.loss, tvars)
        # optimizer = tf.keras.optimizers.Adam(lr=lr)
        # self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))
    def call(self, x, training=False, mask=None):
        # x = self.hidden1(x, training=training)
        # x = self.hidden2(x, training=training)
        #
        # self.chosen_action = tf.argmax(x, 1)  # use maximum value as selected action
        #
        # self.indexes = tf.range(0, x.shape[0]) * x.shape[1] + self.action_holder  # 1 from output.shape[0], 4 from output.shape[1]
        # self.responsible_outputs = tf.gather(tf.reshape(x, [-1]), self.indexes)
        #
        #
        # self.gradient_holders = []
        # for idx, var in enumerate(self.trainable_variables):
        #     placeholder = keras.Input(shape=[None], name=str(idx) + '_holder', dtype=tf.float32)
        #     self.gradient_holders.append(placeholder)
        with tf.GradientTape() as tape:
            x = self.hidden1(x, training=training)
            x = self.hidden2(x, training=training)
            self.chosen_action = tf.argmax(x, 1)  # use maximum value as selected action
            self.indexes = tf.range(0, x.shape[0]) * x.shape[1] + self.action_holder  # 1 from output.shape[0], 4 from output.shape[1]
            self.responsible_outputs = tf.gather(tf.reshape(x, [-1]), self.indexes)
            self.gradient_holders = []

            for idx, var in enumerate(self.trainable_variables):
                placeholder = keras.Input(shape=[None], name=str(idx) + '_holder', dtype=tf.float32)
                self.gradient_holders.append(placeholder)
            # loss = -tf.reduce_mean(tf.math.log(self.responsible_outputs) * self.reward_holder)
            # self.loss = -tf.reduce_mean(tf.math.log(self.responsible_outputs) * self.reward_holder)
        # self.gradients = tape.gradient(loss, self.trainable_variables)
        # self.gradients = tf.gradients(self.loss, self.trainable_variables)
        return x

ops.reset_default_graph() #Clear the Tensorflow graph.

model = SimpleAgent(s_size=4,a_size=2,h_size=8)
optimizer = tf.keras.optimizers.Adam(lr=1e-2)
# myAgent = agent(lr=1e-2,s_size=4,a_size=2,h_size=8) #Load the agent.

total_episodes = 5000 #Set total number of episodes to train agent on.
max_ep = 999
update_frequency = 5

i = 0
total_reward = []
total_length = []
gradBuffer = model.trainable_variables
for ix, grad in enumerate(gradBuffer): gradBuffer[ix] = grad * 0
while i < total_episodes:
    s = env.reset()
    running_reward = 0
    ep_history = []
    for j in range(max_ep):
        #Probabilistically pick an action given our network outputs.
        print(s)
        a_dist = model(s.reshape(-1, 4)).numpy()
        print(a_dist)
        # a_dist /= a_dist.numpy().sum() # avoid "ValueError: probabilities do not sum to 1"
        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = np.argmax(a_dist == a)
        s1, r, d, _ = env.step(a)  # Get our reward for taking an action given a bandit.
        ep_history.append([s, a, r, s1])
        s = s1
        running_reward += r
        if d == True:
            # Update the network.
            ep_history = np.array(ep_history)
            ep_history[:, 2] = discount_rewards(ep_history[:, 2])
            # feed_dict = {model.reward_holder: ep_history[: 2],
            #              model.action_holder: ep_history[:, 1],
            #              model.state_in: np.vstack(ep_history[:, 0])
            #              }
            # grads = model.gradients(feed_dict, model.trainable_variables)
            # for idx, grad in enumerate(grads): gradBuffer[idx] += grad
            if i % update_frequency == 0 and i != 0:
                # feed_dict = dictionary = dict(zip(model.gradient_holders, gradBuffer))
                optimizer.apply_gradients(zip(model.gradient_holders, model.trainable_variables))
                # model.update_batch()    # feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                for ix, grad in enumerate(gradBuffer): gradBuffer[ix] = grad * 0
            total_reward.append(running_reward)
            total_length.append(j)
            break
    if i % 100 == 0:
        print(np.mean(total_reward[-100:]))
    i += 1