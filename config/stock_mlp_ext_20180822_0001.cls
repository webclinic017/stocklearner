[Model]
name = stock_mlp_ext_20180822_0001
type = MLP

[Input]
unit = 8

[Layer1]
unit = 1024
act_fn = tf.nn.relu6

[Layer2]
unit = 512
act_fn = tf.nn.relu6

[Layer3]
unit = 256
act_fn = tf.nn.relu6

[Layer4]
unit = 128
act_fn = tf.nn.relu6

[Layer5]
unit = 64
act_fn = tf.nn.relu6

[Layer6]
unit = 32
act_fn = tf.nn.relu6

[Layer7]
unit = 16
act_fn = tf.nn.relu6

[Output]
unit = 8

[Dataset]
batch_size = 200
repeat_time = -1

[Hyper Parameters]
type = classification
learning_rate = 0.0001
decay_learning_rate = false
decay_rate = 0.9
decay_step = 100
log_dir = D:\Models\mlp\mlp_ext_20180822_0001\
loss_fn = cross_entropy
l2_loss = false
opt_fn = tf.train.AdamOptimizer
acc_fn = correct_prediction
model_dir = D:\Models\mlp\mlp_ext_20180822_0001\
echo = 50000
enable_tensorboard_log = true