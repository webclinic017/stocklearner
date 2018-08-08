[Input]
unit = 28

[RNN Layer]
cell_type = LSTM
forget_bias = 1.0
hidden_cells = 128
time_steps = 28
act_fn = tf.nn.relu6

[Output]
unit = 10

[Dataset]
batch_size = 200
repeat_time = -1

[Hyper Parameters]
type = classification
learning_rate = 0.0001
decay_learning_rate = false
decay_rate = 0.9
decay_step = 100
log_dir = models/
loss_fn = cross_entropy
l2_loss = false
opt_fn = tf.train.AdamOptimizer
acc_fn = correct_prediction
model_dir = models/
echo = 10000