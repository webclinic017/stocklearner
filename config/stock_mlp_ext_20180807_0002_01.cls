[Model]
name = stock_mlp_ext_20180807_0002_01
type = MLP

[Input]
unit = 8

[Layer1]
unit = 50
act_fn = tf.nn.relu6

[Layer2]
unit = 50
act_fn = tf.nn.relu6

[Layer3]
unit = 40
act_fn = tf.nn.relu6

[Layer4]
unit = 40
act_fn = tf.nn.relu6

[Layer5]
unit = 30
act_fn = tf.nn.relu6


[Layer6]
unit = 30
act_fn = tf.nn.relu6

[Layer7]
unit = 20
act_fn = tf.nn.relu6

[Layer8]
unit = 20
act_fn = tf.nn.relu6

[Layer9]
unit = 10
act_fn = tf.nn.relu6

[Layer10]
unit = 10
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
log_dir = D:\Models\mlp\mlp_ext_20180807_0002_01\
loss_fn = cross_entropy
l2_loss = false
opt_fn = tf.train.AdamOptimizer
acc_fn = correct_prediction
model_dir = D:\Models\mlp\mlp_ext_20180807_0002_01\
echo = 50000
enable_tensorboard_log = true