[network]
type = AutoEncoder

[input]
unit = 28
act_fn = tf.nn.relu6

[layer1]
unit = 128
act_fn = tf.nn.relu6

[layer2]
unit = 64
keep_prob = 0.9
act_fn = tf.nn.relu6

[layer3]
unit = 32
keep_prob = 0.9
act_fn = tf.nn.relu6
