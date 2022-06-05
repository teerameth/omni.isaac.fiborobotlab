import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
batch_size = 16
batch_count = 10
patience = 5 # if no improvement in N steps -> stop training

class SimpleLSTM(keras.Model):
    def __init__(self, num_input, embedding_dim, lstm_units, num_output):
        super().__init__(self)
        self.embedding = layers.Embedding(num_input, embedding_dim)
        self.lstm1 = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.dense = layers.Dense(num_output)
    def call(self, inputs, states=None, return_state = False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None: states = self.lstm1.get_initial_state(x)
        x, states = self.lstm1(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state: return x, states
        else: return x

model = SimpleLSTM(num_input=9, embedding_dim=32, lstm_units=64, num_output=3)
model.build(input_shape=(16, 9))
model.summary()