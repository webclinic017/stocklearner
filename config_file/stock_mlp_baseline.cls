[network]
type = MLP

[input]
unit = 8

[layer1]
unit = 50
act_fn = tf.nn.relu6

[layer2]
unit = 40
act_fn = tf.nn.relu6

[layer3]
unit = 30
act_fn = tf.nn.relu6

[layer4]
unit = 20
act_fn = tf.nn.relu6

[layer5]
unit = 10
act_fn = tf.nn.relu6

[output]
unit = 8
act_fn = tf.nn.relu6