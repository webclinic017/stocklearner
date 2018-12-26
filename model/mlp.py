from model.base_network import *

IRIS_BOUNDARIES = [0, 1]


class MLP(Network):
    def __init__(self, config_file, network_name=""):
        Network.__init__(self, config_file)

        self._network_name = network_name

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
            with tf.name_scope(layer):
                n_units = self.config.getint(layer, "unit")

                if layer == "Input":
                    self.x = tf.placeholder(dtype=tf.float32, shape=[None, n_units], name=self._network_name + layer)
                    self.logger.info("Building Input Layer:Input Size =>" + str(n_units))
                    self.network = self.x

                elif layer == "Output":
                    self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, n_units], name=self._network_name + layer)
                    self.network = tf.layers.dense(self.network, n_units, activation=act, name=self._network_name + layer)
                    self.logger.info("Building Input Layer:Output Size =>" + str(n_units))
                else:
                    # TODO：1. modify dropout to use tf.layers.dropout
                    # TODO: 2. add batch normalization
                    act = fn_util.get_act_fn(self.config.get(layer, "act_fn"))
                    self.network = tf.layers.dense(self.network, n_units, activation=act, name=self._network_name + layer)
                    self.logger.info("Building Hidden Layer:Unit Size =>" + str(n_units))

                    if self.config.has_option(layer, "keep_prob"):
                        keep_prob = self.config.getfloat(layer, "keep_prob")
                        self.network = tf.layers.dropout(self.network, rate=(1. - keep_prob), training=True)
                        # self.network = tf.nn.dropout(self.network, keep_prob=keep_prob, name=layer+"_dropout")
                        self.logger.info("Building Dropout Layer:Keep Prob =>" + str(keep_prob))

    def predict(self, batch_x):
        if not os.path.exists(self.model_dir):
            raise ModelNotTrained()

        if len(os.listdir(self.model_dir)) > 0:
            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)

                saver = tf.train.Saver()
                # saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(self.model_dir) + ".meta")
                saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))

                y = sess.run(self.network, feed_dict={self.x: batch_x})
                self.logger.info(y)
                self.logger.info(sess.run(tf.argmax(y, 1)))
