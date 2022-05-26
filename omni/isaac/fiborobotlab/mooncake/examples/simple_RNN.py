import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
batch_size = 16
batch_count = 10
patience = 5 # if no improvement in N steps -> stop training

timesteps = 9
input_dim = 1

# Build the RNN model
def build_model():
    model = keras.models.Sequential(
        [
            keras.layers.LSTM(32, input_shape=(timesteps, input_dim)),  # (time-steps, n_features)
            # keras.layers.BatchNormalization(),
            keras.layers.Dense(1),
        ]
    )
    return model

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
# mnist = keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
# sample, sample_label = x_train[0], y_train[0]
model = build_model()
print(model.summary())
optimizer = tf.optimizers.Adam(learning_rate=0.0025)

# Instantiate a loss function.
# loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # Computes the crossentropy loss between the labels and predictions. (use one-hot) produces a category index of the most likely matching category.
# loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)  # Computes the crossentropy loss between the labels and predictions. (use one-hot) produces a one-hot array containing the probable match for each category.
loss_fn = keras.losses.MeanSquaredError()  # Instantiate a loss function.

# Prepare the metrics.
# train_acc_metric = keras.metrics.CategoricalAccuracy()
# val_acc_metric = keras.metrics.CategoricalAccuracy()
# test_acc_metric = keras.metrics.CategoricalAccuracy()
train_acc_metric = keras.metrics.MeanSquaredError()
val_acc_metric = keras.metrics.MeanSquaredError()
test_acc_metric = keras.metrics.MeanSquaredError()

val_loss_tracker = []
for epoch in range(100):
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
                                train_acc_metric)
        train_loss.append(float(loss_value))
    # Run a validation loop at the end of each epoch
    for step in range(batch_count):
        x_batch_val, y_batch_val = get_val_data()
        val_loss_value = test_step(x_batch_val, y_batch_val,
                                   model, loss_fn,
                                   val_acc_metric)
        val_loss.append(float(val_loss_value))

    val_loss_tracker.append(np.mean(val_loss))
    # Display metrics at the end of each epoch
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))
    val_acc = val_acc_metric.result()
    print("Validation acc: %.4f" % (float(val_acc),))
    test_acc = test_acc_metric.result()
    # Reset metrics at the end of each epoch
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()
    if len(val_loss_tracker) > patience:
        still_better = False
        for i in range(patience):
            if val_loss_tracker[len(val_loss_tracker) - patience + i] < min(
                val_loss_tracker[:len(val_loss_tracker) - patience]): still_better = True
        if still_better == False: break
# model.fit(
#     x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=10
# )