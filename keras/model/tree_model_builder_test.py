import tensorflow as tf
from keras.model.tree_model_builder import TreeModelBuilder

if __name__ == "__main__":
    builder = TreeModelBuilder("../../config_file/yaml_config/mnist_model.yaml")
    keras_model = builder.get_model()
    # estimator = tf.keras.estimator.model_to_estimator(keras_model=keras_model)

    # train, test = tf.keras.datasets.mnist.load_data()
    # mnist_x, mnist_y = train
    #
    # def get_train_inputs(batch_size, mnist_data):
    #     """Return the input function to get the training data.
    #     Args:
    #         batch_size (int): Batch size of training iterator that is returned
    #                           by the input function.
    #         mnist_data ((array, array): Mnist training data as (inputs, labels).
    #     Returns:
    #         DataSet: A tensorflow DataSet object to represent the training input
    #                  pipeline.
    #     """
    #     dataset = tf.data.Dataset.from_tensor_slices(mnist_data)
    #     dataset = dataset.shuffle(
    #         buffer_size=1000, reshuffle_each_iteration=True
    #     ).repeat(count=None).batch(batch_size)
    #     return dataset

    batch_size = 128
    num_classes = 10
    epochs = 20

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    print(y_train)
    print(y_test)

    history = keras_model.fit(x_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=1,
                              validation_data=(x_test, y_test))
    score = keras_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
