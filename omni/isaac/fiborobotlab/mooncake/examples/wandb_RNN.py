import wandb
from wandb.keras import WandbCallback

wandb.login()

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# batch_size = 16
train_batch_count = 20      # number of training batch(s) per epoch
val_batch_count = 10        # number of validation batch(s) per epoch
patience = 10 # if no improvement in N steps -> stop training

# timesteps = 9
input_dim = 1

sweep_config = {
    'method': 'random',  # bayes
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 10
    },
    'parameters': {
        'timesteps': {
            "min": 1,
            "max": 20
        },
        'batch_size': {
            'values': [8, 16, 32, 64]
        },
        'learning_rate':{
            "min": 0.0001,
            "max": 0.1
        },
        'n_lstm': {
            "min": 1,
            "max": 64
        }
    }
}

# Build the RNN model
def build_model(n_lstm, timesteps):
    model = keras.models.Sequential(
        [
            keras.layers.LSTM(n_lstm, input_shape=(timesteps, input_dim)),  # (time-steps, n_features)
            # keras.layers.BatchNormalization(),
            keras.layers.Dense(1),
        ]
    )
    return model

def get_train_data(batch_size, timesteps):
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
def get_val_data(batch_size, timesteps):
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

def train(model,
          optimizer,
          loss_fn,
          train_mse_metric,
          val_mse_metric,
          batch_size,
          timesteps):
    val_loss_tracker = []
    for epoch in range(1000):
        print("\nStart of epoch %d" % (epoch,))
        train_loss = []
        val_loss = []
        # Iterate over the batches of the dataset
        for step in range(train_batch_count):
            x_batch_train, y_batch_train = get_train_data(batch_size=batch_size, timesteps=timesteps)
            loss_value = train_step(x_batch_train,
                                    y_batch_train,
                                    model,
                                    optimizer,
                                    loss_fn,
                                    train_mse_metric)
            train_loss.append(float(loss_value))
        # Run a validation loop at the end of each epoch
        for step in range(val_batch_count):
            x_batch_val, y_batch_val = get_val_data(batch_size=batch_size, timesteps=timesteps)
            val_loss_value = test_step(x_batch_val,
                                       y_batch_val,
                                       model,
                                       loss_fn,
                                       val_mse_metric)
            val_loss.append(float(val_loss_value))
        val_loss_tracker.append(np.mean(val_loss))  # track validation loss for no improvment elimination manually
        # Display metrics at the end of each epoch
        train_mse = train_mse_metric.result()
        print("Training mse over epoch: %.4f" % (float(train_mse),))
        val_mse= val_mse_metric.result()
        print("Validation mse: %.4f" % (float(val_mse),))
        # Reset metrics at the end of each epoch
        train_mse_metric.reset_states()
        val_mse_metric.reset_states()
        # 3️⃣ log metrics using wandb.log
        wandb.log({'epochs': epoch,
                   'loss': np.mean(train_loss),
                   'mse': float(train_mse),
                   'val_loss': np.mean(val_loss),
                   'val_mse': float(val_mse)})
        if len(val_loss_tracker) > patience:
            still_better = False
            for i in range(patience):
                if val_loss_tracker[len(val_loss_tracker) - patience + i] < min(val_loss_tracker[:len(val_loss_tracker) - patience]): still_better = True
            if still_better == False: break

def sweep_train():
    config_defaults = { # default hyperparameters
        'batch_size': 8,
        'learning_rate': 0.01
    }
    # Initialize wandb with a sample project name
    wandb.init(config=config_defaults)  # this gets over-written in the Sweep
    wandb.config.architecture_name = "RNN"
    wandb.config.dataset_name = "cosine_test"

    model = build_model(n_lstm=wandb.config.n_lstm, timesteps=wandb.config.timesteps)
    print(model.summary())
    optimizer = tf.optimizers.Adam(learning_rate=wandb.config.learning_rate)

    # Instantiate a loss function.
    # loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # Computes the crossentropy loss between the labels and predictions. (use one-hot) produces a category index of the most likely matching category.
    # loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)  # Computes the crossentropy loss between the labels and predictions. (use one-hot) produces a one-hot array containing the probable match for each category.
    loss_fn = keras.losses.MeanSquaredError()  # Instantiate a loss function.

    # Prepare the metrics.
    # train_acc_metric = keras.metrics.CategoricalAccuracy()
    # val_acc_metric = keras.metrics.CategoricalAccuracy()
    train_mse_metric = keras.metrics.MeanSquaredError()
    val_mse_metric = keras.metrics.MeanSquaredError()

    train(model, optimizer, loss_fn, train_mse_metric, val_mse_metric, batch_size=wandb.config.batch_size, timesteps=wandb.config.timesteps)

sweep_id = wandb.sweep(sweep_config, project="cosine_RNN_test")
wandb.agent(sweep_id, function=sweep_train, count=20000)
