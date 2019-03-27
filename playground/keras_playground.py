import tensorflow as tf
#
# input_layer = tf.keras.layers.InputLayer(input_shape=[None,64], )
#
# main_input = tf.keras.layers.Input(shape=(100,), dtype='int32', name='main_input')

BOUNDARIES = [-7, -5, -3, 0, 3, 5, 7]

# tf.enable_eager_execution()

# with tf.device("/cpu:0"):
#     x = tf.constant([-7.0, -10.7, -1.1, -2, -3, 0, 0, 7, 8, 9.6, 10.1])
#     x = tf.constant([-10., -9., -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     OUTPUT_SIZE = 8
#     DIVIDE_BY = 3
#     s = tf.shape(x)
#     t = tf.add(x, 11)
#     t = tf.divide(t, DIVIDE_BY)
#     t = tf.round(t)
#     t = tf.cast(t, dtype=tf.int64)
#     one_hot_label = tf.reshape(tf.one_hot(t, OUTPUT_SIZE), tf.concat([s, [OUTPUT_SIZE]], axis=0))
#     print(one_hot_label)

# Instantiate a Keras inception v3 model.
keras_inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights=None)

# Compile model with the optimizer, loss, and metrics you'd like to train with.
keras_inception_v3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                           loss='categorical_crossentropy',
                           metric='accuracy')

# Create an Estimator from the compiled Keras model. Note the initial model
# state of the keras model is preserved in the created Estimator.
est_inception_v3 = tf.keras.estimator.model_to_estimator(keras_model=keras_inception_v3)

# Treat the derived Estimator as you would with any other Estimator.
# First, recover the input name(s) of Keras model, so we can use them as the
# feature column name(s) of the Estimator input function:
keras_inception_v3.input_names  # print out: ['input_1']
keras_inception_v3.summary()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(type(x_train))
print(x_train.shape)
print(x_train)
print(y_train)
# Once we have the input name(s), we can create the input function, for example,
# for input(s) in the format of numpy ndarray:
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input_1": x_train},
    y=y_train,
    num_epochs=1,
    shuffle=False)

# To train, we call Estimator's train function:
est_inception_v3.train(input_fn=train_input_fn, steps=2000)