from model.base_network import *
from util import fn_util


class AutoEncoder(Network):
    def __init__(self, config_file):
        Network.__init__(self, config_file)

        # self.__init_hyper_param()
        self.__init_network()

    def __init_hyper_param(self):
        pass

    def __init_network(self):
        self.layers = self.config.sections()
        self.layers.remove("Model")
        self.layers.remove("Hyper Parameters")
        self.layers.remove("Dataset")

        def add_layer(layer_name):
            with tf.name_scope(layer_name):
                with tf.variable_scope(layer_name):
                    W = tf.get_variable("Weight",
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(),
                                        shape=[n_in, n_units])
                    self.var_summaries(W)
                    b = tf.get_variable("Bias",
                                        dtype=tf.float32,
                                        initializer=tf.constant_initializer(value=0.1),
                                        shape=[n_units])
                    self.var_summaries(b)
                with tf.name_scope("Wx_plus_b"):
                    self.preactivate = tf.matmul(self.network, W) + b
                    tf.summary.histogram("pre_activation", self.preactivate)
                print("Building hidden layers:Name =>" + layer + " Size =>[" + str(n_in) + ", " + str(n_units) + "]")

        def add_act_fn(layer_name):
            act = fn_util.get_act_fn(self.config.get(layer_name, "act_fn"))
            self.network = act(self.preactivate)
            tf.summary.histogram("activation", self.network)
            print("Activation Function =>" + self.config.get(layer_name, "act_fn"))

        for layer in self.layers:
            if layer == "Input":
                with tf.name_scope(layer):
                    input_size = self.config.getint(layer, "unit")
                    self.x = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name=layer)
                    self.network = self.x
                    print("Building Input layer:Input Size =>" + str(input_size))
            else:
                n_in = int(self.network.get_shape()[-1])
                n_units = self.config.getint(layer, "unit")
                add_layer("encode" + layer)
                add_act_fn(layer)

        self.layers.sort(reverse=True)

        for layer in self.layers[1:]:
            n_in = int(self.network.get_shape()[-1])
            n_units = self.config.getint(layer, "unit")
            if layer == "Input":
                with tf.name_scope("Output"):
                    output_size = self.config.getint(layer, "unit")
                    self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, output_size], name=layer)
                    self.network = self.preactivate
                    print("Building Output layer:Output Size =>[" + str(n_in) + ", " + str(output_size) + "]")
                    add_act_fn(layer)
            else:
                add_layer("decode" + layer)
                add_act_fn(layer)

if __name__ == "__main__":
    from feed.csv_data import csv_input_fn

    config_file_path = "../config/mnist_autoencoder_baseline.cls"
    training_data_path = "D:\\Output\\000017\\"
    training_dataset = csv_input_fn(training_data_path)

    ae = AutoEncoder(config_file=config_file_path)
    # AutoEncoder.train(dataset=training_dataset)
