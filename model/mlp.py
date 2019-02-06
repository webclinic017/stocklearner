from model.base_network import *
from util import fn_util

'''
self.x => input
self.y => output
self.y_ => label
self.is_training => training indicator
'''


class MLP(Network):
    def __init__(self, config_file, network_name=""):
        Network.__init__(self, config_file)
        self.network_name = network_name
        self.__init_network()

    def __init_network(self):
        self.layers = self.config.sections()
        self.layers.remove("network")

        with tf.name_scope(self.network_name):
            self.is_training = tf.placeholder(dtype=tf.bool, shape=None, name="is_training")

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
                        # TODO: Testing for batch normalization and dropout
                        self.network = tf.layers.dense(self.network, n_units, activation=None, name= layer + "_Wx_plus_b")
                        self.logger.info("Building Hidden Layer:Unit Size =>" + str(n_units))
                        print("Building Hidden Layer:Unit Size =>" + str(n_units))

                        if self.config.has_option(layer, "batch_normalization"):
                            self.network = tf.layers.batch_normalization(self.network)
                            self.logger.info("Adding Batch Normalization to " + layer)
                            print("Adding Batch Normalization to " + layer)

                        if self.config.has_option(layer, "keep_prob"):
                            keep_prob = self.config.getfloat(layer, "keep_prob")
                            self.network = tf.layers.dropout(self.network, rate=(1. - keep_prob), training=self.is_training)
                            # self.network = tf.nn.dropout(self.network, keep_prob=keep_prob, name=layer+"_dropout")
                            self.logger.info("Adding Dropout Layer:Keep Prob =>" + str(keep_prob))
                            print("Adding Dropout Layer:Keep Prob =>" + str(keep_prob))

                        act = fn_util.get_act_fn(self.config.get(layer, "act_fn"))
                        self.network = act(self.network)
                        self.logger.info("Adding Activation Function to " + layer)
                        print("Adding Activation Function to " + layer)
