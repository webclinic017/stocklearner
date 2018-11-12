from model.base_network import *

MNIST_BOUNDARIES = [0, 1, 2, 3, 4, 5, 6, 7, 8]


class RNN(Network):
    def __init__(self, config_file):
        Network.__init__(self, config_file)

        self._init_hyper_param()
        self._init_network()
        self._add_train_ops()

    def _init_hyper_param(self):
        # add additional hyper parameter if necessary
        pass

    def _init_network(self):
        self.layers = self.config.sections()
        self.layers.remove("Model")
        self.layers.remove("Hyper Parameters")
        self.layers.remove("Dataset")
        self.layers.remove("Input")
        self.layers.remove("Output")

        # num_of_rnn_layers = len(list(filter(lambda x: x[:3] == "RNN", self.layers)))
        # print(self.layers)
        # print(len(list(filter(lambda x: x[:3] == "RNN", self.layers))))

        with tf.name_scope("Input"):
            self.input_size = self.config.getint("Input", "unit")
            self.time_steps = self.config.getint("Input", "time_steps")
            self.x = tf.placeholder("float", [None, self.time_steps, self.input_size])
            self.network = tf.unstack(self.x, self.time_steps, 1)
            print("Building Input Layer:Input Size =>" + str(self.input_size))
            print("Building Input Layer:Time Steps =>" + str(self.time_steps))

        stacked_rnn = []
        for layer in self.layers:
            print("Building for " + layer)
            cell_type = self.config.get(layer, "cell_type")
            print("Building RNN Cells:Cell Type =>" + cell_type)

            hidden_cells = self.config.getint(layer, "hidden_cells")
            forget_bias = self.config.getfloat(layer, "forget_bias")
            print("Building RNN Cells:Hidden Cells =>" + str(hidden_cells))
            layer_cell = dl_util.get_rnn_cells(cell_type, hidden_cells, forget_bias)

            stacked_rnn.append(layer_cell)

        if len(self.layers) > 1:
            cell = tf.contrib.rnn.MultiRNNCell(stacked_rnn)
            print("This is a Multi RNN Layer")
        else:
            cell = stacked_rnn[0]
            print("This is a Single RNN Layer")

        self.network, self.states = tf.contrib.rnn.static_rnn(cell, self.network, dtype=tf.float32)
        print("Using Static RNN")

        with tf.name_scope("Output"):
            self.output_size = self.config.getint("Output", "unit")
            self.y_ = tf.placeholder("float", [None, self.output_size])
            act = fn_util.get_act_fn(self.config.get("Output", "act_fn"))
            self.network = tf.contrib.layers.fully_connected(self.network[-1], self.output_size, activation_fn=act)
            print("Building Output Layer:Output Size =>" + str(self.output_size))

    def eval(self, dataset):
        pass

    def predict(self, batch_x):
        pass
