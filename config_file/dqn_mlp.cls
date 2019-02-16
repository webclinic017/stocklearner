[network]
type = MLP

[input]
unit = 8

[layer1]
unit = 512
act_fn = tf.nn.relu6
keep_prob = 0.9
batch_normalization = true

[layer2]
unit = 256
act_fn = tf.nn.relu6
keep_prob = 0.9
batch_normalization = true

[layer3]
unit = 128
act_fn = tf.nn.relu6
keep_prob = 0.9
batch_normalization = true

[layer4]
unit = 64
act_fn = tf.nn.relu6
keep_prob = 0.9
batch_normalization = true

[output]
unit = 4
act_fn = tf.nn.relu6