from model.base_network import *
from util import fn_util
from util import dl_util

MNIST_BOUNDARIES = [0, 1, 2, 3, 4, 5, 6, 7, 8]


class RNN(Network):
    def __init__(self, config_file, network_name=""):
        Network.__init__(self, config_file)
        self.network_name = network_name
        self.__init_network()

    def __init_network(self):
        self.layers = self.config.sections()
        self.layers.remove("network")
        self.layers.remove("input")
        self.layers.remove("output")

        # num_of_rnn_layers = len(list(filter(lambda x: x[:3] == "RNN", self.layers)))
        # self.logger.info(self.layers)
        # self.logger.info(len(list(filter(lambda x: x[:3] == "RNN", self.layers))))

        with tf.name_scope("input"):
            self.input_size = self.config.getint("input", "unit")
            self.time_steps = self.config.getint("input", "time_steps")
            self.x = tf.placeholder("float", [None, self.time_steps, self.input_size])
            self.network = tf.unstack(self.x, self.time_steps, 1)
            self.logger.info("Building Input Layer:Input Size =>" + str(self.input_size))
            self.logger.info("Building Input Layer:Time Steps =>" + str(self.time_steps))
            print("Building Input Layer:Input Size =>" + str(self.input_size))
            print("Building Input Layer:Time Steps =>" + str(self.time_steps))

        stacked_rnn = []
        for layer in self.layers:
            self.logger.info("Building for " + layer)
            print("Building for " + layer)
            cell_type = self.config.get(layer, "cell_type")
            self.logger.info("Building RNN Cells:Cell Type =>" + cell_type)
            print("Building RNN Cells:Cell Type =>" + cell_type)

            hidden_cells = self.config.getint(layer, "hidden_cells")
            forget_bias = self.config.getfloat(layer, "forget_bias")
            self.logger.info("Building RNN Cells:Hidden Cells =>" + str(hidden_cells))
            print("Building RNN Cells:Hidden Cells =>" + str(hidden_cells))
            layer_cell = dl_util.get_rnn_cells(cell_type, hidden_cells, forget_bias)

            stacked_rnn.append(layer_cell)

        if len(self.layers) > 1:
            cell = tf.contrib.rnn.MultiRNNCell(stacked_rnn)
            self.logger.info("This is a Multi RNN Layer")
            print("This is a Multi RNN Layer")
        else:
            cell = stacked_rnn[0]
            self.logger.info("This is a Single RNN Layer")
            print("This is Single RNN Layer")

        self.network, self.states = tf.contrib.rnn.static_rnn(cell, self.network, dtype=tf.float32)
        self.logger.info("Using Static RNN")
        print("Using Static RNN")

        with tf.name_scope("output"):
            self.output_size = self.config.getint("output", "unit")
            self.y_ = tf.placeholder("float", [None, self.output_size])
            act = fn_util.get_act_fn(self.config.get("output", "act_fn"))
            self.y = tf.contrib.layers.fully_connected(self.network[-1], self.output_size, activation_fn=act)
            self.logger.info("Building Output Layer:Output Size =>" + str(self.output_size))
            print("Building Output Layer:Output Size =>" + str(self.output_size))
