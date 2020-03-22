from util import fn_util


class AutoEncoder(Network):
    def __init__(self, config_file, network_name=""):
        self.network = network_name
        Network.__init__(self, config_file)

        self.__init_network()

    def __init_network(self):
        self.layers = self.config.sections()
        self.layers.remove("network")

        for layer in self.layers:
            n_units = self.config.getint(layer, "unit")
            if layer == "input":
                self.x = tf.placeholder(dtype=tf.float32, shape=[None, n_units], name=layer)
                self.network = self.x
                self.logger.info("Building Input layer:Input Size =>" + str(n_units))
                print("Building Input layer:Input Size =>" + str(n_units))
            else:
                act = fn_util.get_act_fn(self.config.get(layer, "act_fn"))
                self.network = tf.layers.dense(self.network, n_units, activation=act, name="encode_" + layer)
                self.logger.info("Building Encode " + layer + ":Unit Size =>" + str(n_units))
                print("Building Encode " + layer + ":Unit Size =>" + str(n_units))

                if self.config.has_option(layer, "keep_prob"):
                    keep_prob = self.config.getfloat(layer, "keep_prob")
                    self.network = tf.nn.dropout(self.network, keep_prob=keep_prob, name="encode_" + layer + "_dropout")
                    self.logger.info("Building Dropout " + layer + ":Keep Prob =>" + str(keep_prob))
                    print("Building Dropout " + layer + ":Keep Prob =>" + str(keep_prob))

        self.layers.sort(reverse=True)

        for layer in self.layers[1:]:
            n_units = self.config.getint(layer, "unit")
            if layer == "input":
                self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, n_units], name="output")
                act = fn_util.get_act_fn(self.config.get(layer, "act_fn"))
                self.y = tf.layers.dense(self.network, n_units, activation=act, name=layer)
                self.logger.info("Building Output " + layer + ":Output Size =>" + str(n_units))
                print("Building Output " + layer + ":Output Size =>" + str(n_units))
            else:
                act = fn_util.get_act_fn(self.config.get(layer, "act_fn"))
                self.network = tf.layers.dense(self.network, n_units, activation=act, name="decode_" + layer)
                self.logger.info("Building Decode " + layer + ":Unit Size =>" + str(n_units))
                print("Building Decode " + layer + ":Unit Size =>" + str(n_units))

                if self.config.has_option(layer, "keep_prob"):
                    keep_prob = self.config.getfloat(layer, "keep_prob")
                    self.network = tf.nn.dropout(self.network, keep_prob=keep_prob, name="decode_" + layer + "_dropout")
                    self.logger.info("Building Dropout " + layer + ":Keep Prob =>" + str(keep_prob))
                    print("Building Dropout " + layer + ":Keep Prob =>" + str(keep_prob))
