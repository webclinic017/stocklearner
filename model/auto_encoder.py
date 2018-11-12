from model.base_network import *
from util import fn_util


class AutoEncoder(Network):
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

        for layer in self.layers:
            n_units = self.config.getint(layer, "unit")
            if layer == "Input":
                self.x = tf.placeholder(dtype=tf.float32, shape=[None, n_units], name=layer)
                self.network = self.x
                print("Building Input layer:Input Size =>" + str(n_units))
            else:
                act = fn_util.get_act_fn(self.config.get(layer, "act_fn"))
                self.network = tf.layers.dense(self.network, n_units, activation=act, name="encode_" + layer)
                print("Building Encode " + layer + ":Unit Size =>" + str(n_units))

                if self.config.has_option(layer, "keep_prob"):
                    keep_prob = self.config.getfloat(layer, "keep_prob")
                    self.network = tf.nn.dropout(self.network, keep_prob=keep_prob, name="encode_" + layer + "_dropout")
                    print("Building Dropout " + layer + ":Keep Prob =>" + str(keep_prob))

        self.layers.sort(reverse=True)

        for layer in self.layers[1:]:
            n_units = self.config.getint(layer, "unit")
            if layer == "Input":
                self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, n_units], name="Output")
                act = fn_util.get_act_fn(self.config.get(layer, "act_fn"))
                self.network = tf.layers.dense(self.network, n_units, activation=act, name=layer)
                print("Building Output " + layer + ":Output Size =>" + str(n_units))
            else:
                act = fn_util.get_act_fn(self.config.get(layer, "act_fn"))
                self.network = tf.layers.dense(self.network, n_units, activation=act, name="decode_" + layer)
                print("Building Decode " + layer + ":Unit Size =>" + str(n_units))

                if self.config.has_option(layer, "keep_prob"):
                    keep_prob = self.config.getfloat(layer, "keep_prob")
                    self.network = tf.nn.dropout(self.network, keep_prob=keep_prob, name="decode_" + layer + "_dropout")
                    print("Building Dropout " + layer + ":Keep Prob =>" + str(keep_prob))
