import datetime
from util import dl_util
from model.base_network import *

IRIS_BOUNDARIES = [0, 1]


class MLP(Network):
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
            with tf.name_scope(layer):
                n_units = self.config.getint(layer, "unit")

                if layer == "Input":
                    self.x = tf.placeholder(dtype=tf.float32, shape=[None, n_units], name=layer)
                    print("Building Input Layer:Input Size =>" + str(n_units))
                    self.network = self.x

                elif layer == "Output":
                    self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, n_units], name=layer)
                    self.network = tf.layers.dense(self.network, n_units, activation=act, name=layer)
                    print("Building Input Layer:Output Size =>" + str(n_units))
                else:
                    act = fn_util.get_act_fn(self.config.get(layer, "act_fn"))
                    self.network = tf.layers.dense(self.network, n_units, activation=act, name=layer)
                    print("Building Hidden Layer:Unit Size =>" + str(n_units))

                    if self.config.has_option(layer, "keep_prob"):
                        keep_prob = self.config.getfloat(layer, "keep_prob")
                        self.network = tf.nn.dropout(self.network, keep_prob=keep_prob, name=layer+"_dropout")
                        print("Building Dropout Layer:Keep Prob =>" + str(keep_prob))

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
