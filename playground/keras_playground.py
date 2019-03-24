import tensorflow as tf
#
# input_layer = tf.keras.layers.InputLayer(input_shape=[None,64], )
#
# main_input = tf.keras.layers.Input(shape=(100,), dtype='int32', name='main_input')

BOUNDARIES = [-7, -5, -3, 0, 3, 5, 7]

tf.enable_eager_execution()

x = tf.constant([-1, -2, -3, 0, 0, 7, 8])
t = tf.add(x, 10)
print(t)
print(tf.one_hot(t, 22))
