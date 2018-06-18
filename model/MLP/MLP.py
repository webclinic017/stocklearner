import configparser
import tensorflow as tf


class MLP:
    def __init__(self, config_file):
        self.__config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(self.__config_file)

        self.__init_hyper_param()
        self.__init_network()

    def __init_hyper_param(self):
        self.learning_rate = self.config.getfloat('Hyper Parameters', 'learning_rate')
        self.batch_size = self.config.getint('Hyper Parameters', 'batch_size')
        self.type = self.config.get('Hyper Parameters', 'type')
        self.log_dir = self.config.get('Hyper Parameters', 'log_dir')
        self.loss_fn = self.config.get('Hyper Parameters', 'loss_fn')
        self.opt_fn = self.config.get('Hyper Parameters', 'opt_fn')
        self.acc_fn = self.config.get('Hyper Parameters', 'acc_fn')
        self.model_dir = self.config.get('Hyper Parameters', 'model_dir')

        if self.type == "classification":
            self.one_hot = True
        else:
            self.one_hot = False

    @staticmethod
    def __var_summaries(var):
        with tf.name_scope('Summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def __init_network(self):
        self.layers = self.config.sections()
        self.layers.remove('Hyper Parameters')

        for layer in self.layers:
            with tf.name_scope(layer):
                if layer == 'Input':
                    input_size = self.config.getint(layer, 'unit')
                    self.x = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name=layer)
                    self.network = self.x
                    print("Building Input layer:Input Size =>" + str(input_size))
                else:
                    n_in = int(self.network.get_shape()[-1])
                    n_units = self.config.getint(layer, 'unit')
                    W = tf.Variable(tf.truncated_normal([n_in, n_units], stddev=0.1))
                    self.__var_summaries(W)
                    b = tf.Variable(tf.constant(0.1, shape=[n_units]))
                    self.__var_summaries(b)
                    with tf.name_scope('Wx_plus_b'):
                        preactivate = tf.matmul(self.network, W) + b
                        tf.summary.histogram('pre_activation', preactivate)
                    if layer == 'Output':
                        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, n_units], name=layer)
                        self.network = preactivate
                        print("Building Output layer:Output Size =>[" + str(n_in) + ", " + str(n_units) + "]")
                    else:
                        act = fn_util.get_act_fn(self.config.get(layer, 'act_fn'))
                        self.network = act(preactivate)
                        tf.summary.histogram('activation', self.network)
                        print("Building hidden layers:Name =>" + layer + " Size =>[" + str(n_in) + ", " + str(n_units) + "]")
                        print("Activation Function =>" + self.config.get(layer, 'act_fn'))
                        try:
                            with tf.name_scope('dropout'):
                                keep_prob = self.config.getfloat(layer, 'keep_prob')
                                self.network = tf.nn.dropout(self.network, keep_prob=keep_prob)
                                tf.summary.scalar('dropout', keep_prob)
                                print("Keep prob =>" + str(keep_prob))
                        except Exception as ex:
                            pass

        with tf.name_scope('Loss'):
            with tf.name_scope(self.loss_fn):
                self.cost = fn_util.get_loss_fn(self.loss_fn, self.y_, self.network)
                tf.summary.scalar(self.loss_fn, self.cost)
        with tf.name_scope('Train_Step'):
            optimizer = fn_util.get_opt_fn(self.opt_fn)
            self.train_step = optimizer(self.learning_rate).minimize(self.cost)

        with tf.name_scope('Accuracy'):
            self.accuracy = fn_util.get_acc_fn(self.acc_fn, self.y_, self.network)

    def train(self, data_feed):
        with tf.Session() as sess:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.log_dir + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(self.log_dir + '/test')

            saver = tf.train.Saver()

            tf.global_variables_initializer().run()

            # for i in range(self.num_epochs):
            try:
                i = 0
                # while True:
                while i < EPOCH:
                    if i % 10 == 0:
                        test_xs, test_ys = data_feed.get_test_batch(one_hot=self.one_hot)
                        summary, acc = sess.run([merged, self.accuracy], feed_dict={self.x: test_xs, self.y_: test_ys})
                        tf.summary.scalar('accuracy', self.accuracy)
                        print("After epoch " + str(i) + " current accuracy = " + str(acc))
                        test_writer.add_summary(summary, i)
                    else:
                        batch_xs, batch_ys = data_feed.get_next_train_batch(batch_size=self.batch_size,
                                                                            one_hot=self.one_hot)
                        if i % 100 == 99:
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()
                            summary, _ = sess.run([merged, self.train_step],
                                                  feed_dict={self.x: batch_xs, self.y_: batch_ys},
                                                  options=run_options,
                                                  run_metadata=run_metadata)
                            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                            train_writer.add_summary(summary, i)

                            saver.save(sess, self.model_dir + 'MLP_MODEL_' + self.type + '.ckpt')
                        else:
                            summary, _ = sess.run([merged, self.train_step],
                                                  feed_dict={self.x: batch_xs, self.y_: batch_ys})
                            train_writer.add_summary(summary, i)
                    i += 1
            except Exception as _:
                print("End of Training")
                traceback.print_exc()
            train_writer.close()
            test_writer.close()