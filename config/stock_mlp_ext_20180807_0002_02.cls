[Input]
unit = 8

[Layer1]
unit = 50
act_fn = tf.nn.relu6

[Layer2]
unit = 50
act_fn = tf.nn.relu6

[Layer3]
unit = 45
act_fn = tf.nn.relu6

[Layer4]
unit = 45
act_fn = tf.nn.relu6

[Layer5]
unit = 40
act_fn = tf.nn.relu6

[Layer6]
unit = 40
act_fn = tf.nn.relu6

[Layer7]
unit = 35
act_fn = tf.nn.relu6

[Layer8]
unit = 35
act_fn = tf.nn.relu6

[Layer9]
unit = 30
act_fn = tf.nn.relu6

[Layer10]
unit = 30
act_fn = tf.nn.relu6

[Layer11]
unit = 25
act_fn = tf.nn.relu6

[Layer12]
unit = 25
act_fn = tf.nn.relu6

[Layer13]
unit = 20
act_fn = tf.nn.relu6

[Layer14]
unit = 20
act_fn = tf.nn.relu6

[Layer15]
unit = 15
act_fn = tf.nn.relu6

[Layer16]
unit = 15
act_fn = tf.nn.relu6

[Layer17]
unit = 10
act_fn = tf.nn.relu6

[Layer18]
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
log_dir = D:\Models\mlp\mlp_ext_20180807_0002_02\
loss_fn = cross_entropy
l2_loss = false
opt_fn = tf.train.AdamOptimizer
acc_fn = correct_prediction
model_dir = D:\Models\mlp\mlp_ext_20180807_0002_02\
echo = 50000
