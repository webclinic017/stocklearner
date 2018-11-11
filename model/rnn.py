import datetime
from util import dl_util
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

    def train(self, dataset):
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.time_steps * self.batch_size))
        # dataset = dataset.batch(self.batch_size * self.time_steps, drop_remainder= True) # available in tf 1.10
        dataset = dataset.repeat(self.repeat)
        iterator = dataset.make_one_shot_iterator()

        # 2018-08-06 - To fully utilize CPUs for training
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
                # batch_xs = raw_xs # No need to use dict_to_list because MNIST is ndarry
                batch_xs = dl_util.dict_to_list(raw_xs)
                batch_xs = batch_xs.reshape((-1, self.time_steps, self.input_size))
                # batch_ys = dl_util.one_hot(raw_ys, boundaries=MNIST_BOUNDARIES)
                batch_ys = dl_util.one_hot(raw_ys)
                batch_ys = dl_util.rnn_output_split(batch_ys, self.time_steps, self.output_size)

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
                    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
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
        pass

    def predict(self, batch_x):
        pass
