[network]
type = RNN

[input]
unit = 8
time_steps = 10

[rnn_layer1]
cell_type = LSTM
forget_bias = 1.0
hidden_cells = 128

[output]
unit = 8
act_fn = tf.nn.relu6