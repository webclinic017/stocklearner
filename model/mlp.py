import datetime
from util import fn_util
from util import dl_util
from model.base_network import *

IRIS_BOUNDARIES = [0, 1]


class MLP(Network):
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

        for layer in self.layers:
            with tf.name_scope(layer):
                if layer == "Input":
                    input_size = self.config.getint(layer, "unit")
                    self.x = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name=layer)
                    self.network = self.x
                    print("Building Input layer:Input Size =>" + str(input_size))
                else:
                    n_in = int(self.network.get_shape()[-1])
                    n_units = self.config.getint(layer, "unit")
                    with tf.variable_scope(layer):
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
                        preactivate = tf.matmul(self.network, W) + b
                        tf.summary.histogram("pre_activation", preactivate)
                    print("Building hidden layers:Name =>" + layer + " Size =>[" + str(n_in) + ", " + str(n_units) + "]")

                    if layer == "Output":
                        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, n_units], name=layer)
                        self.network = preactivate
                        print("Building Output layer:Output Size =>[" + str(n_in) + ", " + str(n_units) + "]")

                    act = fn_util.get_act_fn(self.config.get(layer, "act_fn"))
                    self.network = act(preactivate)
                    tf.summary.histogram("activation", self.network)
                    print("Activation Function =>" + self.config.get(layer, "act_fn"))
                    try:
                        with tf.name_scope("dropout"):
                            keep_prob = self.config.getfloat(layer, "keep_prob")
                            self.network = tf.nn.dropout(self.network, keep_prob=keep_prob)
                            tf.summary.scalar("dropout", keep_prob)
                            print("Keep prob =>" + str(keep_prob))
                    except Exception as _:
                        pass

        with tf.name_scope("Loss"):
            with tf.name_scope(self.loss_fn):
                self.cost = fn_util.get_loss_fn(self.loss_fn, self.y_, self.network)
                tf.summary.scalar(self.loss_fn, self.cost)
        with tf.name_scope("Train_Step"):
            optimizer = fn_util.get_opt_fn(self.opt_fn)
            self.train_step = optimizer(self.learning_rate).minimize(self.cost)

        with tf.name_scope("Accuracy"):
            self.accuracy = fn_util.get_acc_fn(self.acc_fn, self.y_, self.network)

    def train(self, dataset):
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat(self.repeat)
        iterator = dataset.make_one_shot_iterator()

        # 2018/08/06 - To fully utilize CPUs for training
        config = tf.ConfigProto(device_count={"CPU": 8},
                                inter_op_parallelism_threads=16,
                                intra_op_parallelism_threads=16,
                                log_device_placement=True
                                )

        with tf.Session(config=config) as sess:
            min_cost = 100
            best_accuracy = 0
            avg_accuracy = 0
            merged = tf.summary.merge_all()
            saver = tf.train.Saver(max_to_keep=5)

            if self.tensorboard_summary_enabled:
                writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            next_xs, next_ys = iterator.get_next()

            init = tf.global_variables_initializer()
            sess.run(init)

            global_step = 0
            while global_step < self.echo:
                raw_xs, raw_ys = sess.run([next_xs, next_ys])
                batch_xs = dl_util.dict_to_list(raw_xs)
                batch_ys = dl_util.one_hot(raw_ys)

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _, cost, accuracy = sess.run([merged, self.train_step, self.cost, self.accuracy],
                                                      feed_dict={self.x: batch_xs, self.y_: batch_ys},
                                                      options=run_options,
                                                      run_metadata=run_metadata)

                if self.tensorboard_summary_enabled:
                    writer.add_summary(summary, global_step)

                if min_cost > cost:
                    saver.save(sess, self.model_dir + self.model_name, global_step=global_step)
                    min_cost = cost

                if best_accuracy <= accuracy:
                    best_accuracy = accuracy

                if global_step > 10000:
                    avg_accuracy = avg_accuracy + accuracy

                if global_step % 100 == 0:
                    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    print("Step " + str(global_step) + ": cost is " + str(cost))
                    self.logger.info("Step " + str(global_step) + ": cost is " + str(cost))
                    _, acc = sess.run([merged, self.accuracy], feed_dict={self.x: batch_xs, self.y_: batch_ys})
                    print("Accuracy is: " + str(acc))
                    self.logger.info("Accuracy is: " + str(acc))
                global_step = global_step + 1

            if self.tensorboard_summary_enabled:
                writer.close()

            print("----------------------------------------------------------")
            print("Best Accuracy is: " + str(best_accuracy))
            self.logger.info("Best Accuracy is: " + str(best_accuracy))
            print("Average Accuracy is: " + str(round(avg_accuracy / (self.echo - 10000), 2)))
            self.logger.info("Average Accuracy is: " + str(round(avg_accuracy / (self.echo - 10000), 2)))

    def eval(self, dataset):
        if not os.path.exists(self.model_dir):
            raise ModelNotTrained()

        if len(os.listdir(self.model_dir)) > 0:
            print("----------------------------------------------------------")

            dataset = dataset.batch(self.batch_size)
            dataset = dataset.repeat(self.repeat)
            iterator = dataset.make_one_shot_iterator()

            with tf.Session() as sess:
                next_xs, next_ys = iterator.get_next()

                init = tf.global_variables_initializer()
                sess.run(init)

                saver = tf.train.Saver()
                # saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(self.model_dir)+".meta")
                # print(tf.train.latest_checkpoint(self.model_dir))
                saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))

                raw_xs, raw_ys = sess.run([next_xs, next_ys])
                batch_xs = dl_util.dict_to_list(raw_xs)
                batch_ys = dl_util.one_hot(raw_ys)

                acc = sess.run(self.accuracy, feed_dict={self.x: batch_xs, self.y_: batch_ys})
                print("Accuracy for evaluation is: " + str(acc))
        else:
            raise ModelNotTrained()

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
                print(y)
                print(sess.run(tf.argmax(y, 1)))
