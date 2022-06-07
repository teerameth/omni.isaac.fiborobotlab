import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

input_dim =	3
output_dim = 3
num_timesteps =	2
batch_size = 10
nodes =	10

input_layer = tf.keras.Input(shape=(num_timesteps, input_dim), batch_size=batch_size)

cell = tf.keras.layers.LSTMCell(
    nodes,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='glorot_uniform',
    bias_initializer='zeros',
)

lstm = tf.keras.layers.RNN(
    cell,
    return_state=True,
    return_sequences=True,
    stateful=True,
)

lstm_out, hidden_state, cell_state = lstm(input_layer)

output = tf.keras.layers.Dense(output_dim)(lstm_out)

mdl = tf.keras.Model(
    inputs=input_layer,
    outputs=[hidden_state, cell_state, output]
)
# We can now test what’s going on by passing a batch through the network (look Ma, no tf.Session!):
x = np.random.rand(batch_size, num_timesteps, input_dim).astype(np.float32)
h_state, c_state, out = mdl(x)
print(np.mean(out))
# If we pass this same batch again, we get different result as the hidden state has been changed:
h_state, c_state, out = mdl(x)
print(np.mean(out))
# If we reset the hidden state, we can recover our initial output:
lstm.reset_states(states=[np.zeros((batch_size, nodes)), np.zeros((batch_size, nodes))])
h_state, c_state, out = mdl(x)
print(np.mean(out))
# This method also allows us to use other values than all zeros for the hidden state:
lstm.reset_states(states=[np.ones((batch_size, nodes)), np.ones((batch_size, nodes))])
h_state, c_state, out = mdl(x)
print(np.mean(out))