from model.base_network import *
from util import fn_util

'''
self.x => input
self.y => output
self.y_ => label
'''


class MLP(Network):
    def __init__(self, config_file, network_name=""):
        Network.__init__(self, config_file)
        self.network_name = network_name
        self.__init_network()

    def __init_network(self):
        self.layers = self.config.sections()
        self.layers.remove("model")

        with tf.name_scope(self.network_name):
            for layer in self.layers:
                with tf.name_scope(layer):
                    n_units = self.config.getint(layer, "unit")

                    if layer == "input":
                        self.x = tf.placeholder(dtype=tf.float32, shape=[None, n_units], name=self.network_name + "_" + layer)
                        self.logger.info("Building Input Layer:Input Size =>" + str(n_units))
                        print("Building Input Layer:Input Size =>" + str(n_units))
                        self.network = self.x

                    elif layer == "output":
                        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, n_units], name=self.network_name + "_" + layer)
                        self.y = tf.layers.dense(self.network, n_units, activation=act, name=self.network_name + "_" + layer)
                        self.logger.info("Building Input Layer:Output Size =>" + str(n_units))
                        print("Building Input Layer:Output Size =>" + str(n_units))
                    else:
                        # TODO：1. modify dropout to use tf.layers.dropout
                        # TODO: 2. add batch normalization
                        act = fn_util.get_act_fn(self.config.get(layer, "act_fn"))
                        self.network = tf.layers.dense(self.network, n_units, activation=act, name=self.network_name + "_" + layer)
                        self.logger.info("Building Hidden Layer:Unit Size =>" + str(n_units))
                        print("Building Hidden Layer:Unit Size =>" + str(n_units))

                        if self.config.has_option(layer, "keep_prob"):
                            keep_prob = self.config.getfloat(layer, "keep_prob")
                            self.network = tf.layers.dropout(self.network, rate=(1. - keep_prob), training=True)
                            # self.network = tf.nn.dropout(self.network, keep_prob=keep_prob, name=layer+"_dropout")
                            self.logger.info("Building Dropout Layer:Keep Prob =>" + str(keep_prob))
                            print("Building Dropout Layer:Keep Prob =>" + str(keep_prob))
