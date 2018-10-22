[Model]
name = stock_rnn_baseline
type = RNN

[Input]
unit = 8
time_steps = 10

[RNN_Layer1]
cell_type = LSTM
forget_bias = 1.0
hidden_cells = 128

[Output]
unit = 8
act_fn = tf.nn.relu6

[Dataset]
batch_size = 20
repeat_time = -1

[Hyper Parameters]
type = classification
learning_rate = 0.0001
decay_learning_rate = false
decay_rate = 0.9
decay_step = 100
log_dir = rnn/
loss_fn = cross_entropy
l2_loss = false
opt_fn = tf.train.AdamOptimizer
acc_fn = correct_prediction
model_dir = rnn/
echo = 50000
enable_tensorboard_log = true