[network]
type = MLP

[input]
unit = 8

[layer1]
unit = 50
act_fn = tf.nn.relu6
keep_prob = 0.9
batch_normalization = true

[layer2]
unit = 40
act_fn = tf.nn.relu6
keep_prob = 0.9
batch_normalization = true

[layer3]
unit = 30
act_fn = tf.nn.relu6
keep_prob = 0.9
batch_normalization = true

[layer4]
unit = 20
act_fn = tf.nn.relu6
keep_prob = 0.9
batch_normalization = False

[layer5]
unit = 10
act_fn = tf.nn.relu6
keep_prob = 0.9
batch_normalization = False

[output]
unit = 8
act_fn = tf.nn.relu6