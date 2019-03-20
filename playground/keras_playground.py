import tensorflow as tf

input_layer = tf.keras.layers.InputLayer(input_shape=[None,64], )

main_input = tf.keras.layers.Input(shape=(100,), dtype='int32', name='main_input')