import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
batch_size = 16
batch_count = 10
patience = 5 # if no improvement in N steps -> stop training

timesteps = 9
def get_train_data():
    x_batch_train, y_batch_train = [], []
    for i in range(batch_size):
        offset = random.random()
        width = random.random()*3
        sequence = np.cos(np.arange(offset, offset+width, width/(timesteps+1)))
        x_batch_train.append(sequence[:timesteps])
        y_batch_train.append((sequence[timesteps]+1)/2)
    x_batch_train = np.array(x_batch_train).reshape((batch_size, timesteps, 1))
    y_batch_train = np.array(y_batch_train).reshape((batch_size, 1))
    return x_batch_train, y_batch_train
def get_val_data():
    x_batch_val, y_batch_val = [], []
    for i in range(batch_size):
        offset = i/batch_size
        width = (1+i)/batch_size*3
        sequence = np.cos(np.arange(offset, offset+width, width/(timesteps+1)))
        x_batch_val.append(sequence[:timesteps])
        y_batch_val.append((sequence[timesteps]+1)/2)
    x_batch_val = np.array(x_batch_val).reshape((batch_size, timesteps, 1))
    y_batch_val = np.array(y_batch_val).reshape((batch_size, 1))
    return x_batch_val, y_batch_val

def train_step(x, y, model, optimizer, loss_fn, train_acc_metric):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)  # update training matric
    return loss_value

def test_step(x, y, model, loss_fn, val_acc_metric):
    val_logits = model(x, training=False)
    loss_value = loss_fn(y, val_logits)
    val_acc_metric.update_state(y, val_logits)
    return loss_value

class SimpleLSTM(keras.Model):
    def __init__(self, num_input, embedding_dim, lstm_units, num_output):
        super().__init__(self)
        # self.embedding = layers.Embedding(num_input, embedding_dim)
        self.lstm1 = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.dense = layers.Dense(num_output)
    def call(self, inputs, states=None, return_state = False, training=False):
        x = inputs
        # x = self.embedding(x, training=training)
        if states is None: states = self.lstm1.get_initial_state(x) # state shape = (2, batch_size, lstm_units)
        print(x.shape)
        print(len(states))
        print(states[0].shape)
        x, sequence, states = self.lstm1(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state: return x, states
        else: return x

model = SimpleLSTM(num_input=9, embedding_dim=32, lstm_units=64, num_output=3)
model.build(input_shape=(1, 9))
model.summary()

optimizer = tf.optimizers.Adam(learning_rate=0.0025)
loss_fn = keras.losses.MeanSquaredError()  # Instantiate a loss function.
train_mse_metric = keras.metrics.MeanSquaredError()
val_mse_metric = keras.metrics.MeanSquaredError()
test_mse_metric = keras.metrics.MeanSquaredError()

val_loss_tracker = []
for epoch in range(1000):
    print("\nStart of epoch %d" % (epoch,))
    train_loss = []
    val_loss = []
    test_loss = []
# Iterate over the batches of the dataset
    for step in range(batch_count):
        x_batch_train, y_batch_train = get_train_data()
        loss_value = train_step(x_batch_train,
                                y_batch_train,
                                model,
                                optimizer,
                                loss_fn,
                                train_mse_metric)
        train_loss.append(float(loss_value))
    # Run a validation loop at the end of each epoch
    for step in range(batch_count):
        x_batch_val, y_batch_val = get_val_data()
        val_loss_value = test_step(x_batch_val, y_batch_val,
                                   model, loss_fn,
                                   val_mse_metric)
        val_loss.append(float(val_loss_value))

    val_loss_tracker.append(np.mean(val_loss))
    # Display metrics at the end of each epoch
    train_acc = train_mse_metric.result()
    print("Training mse over epoch: %.4f" % (float(train_acc),))
    val_acc = val_mse_metric.result()
    print("Validation mse: %.4f" % (float(val_acc),))
    test_acc = test_mse_metric.result()
    # Reset metrics at the end of each epoch
    train_mse_metric.reset_states()
    val_mse_metric.reset_states()
    if len(val_loss_tracker) > patience:
        still_better = False
        for i in range(patience):
            if val_loss_tracker[len(val_loss_tracker) - patience + i] < min(
                val_loss_tracker[:len(val_loss_tracker) - patience]): still_better = True
        if still_better == False: break