import configparser
import datetime
import tensorflow as tf
from util import fn_util
from util import dl_util


class RNN:
    def __init__(self, config_file, model_name):
        self.__model_name = model_name
        self.__config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(self.__config_file)

        self.__init_hyper_param()
        self.__init_network()

    def __init_hyper_param(self):
        self.batch_size = self.config.getint("Dataset", "batch_size")
        self.repeat = self.config.getint("Dataset", "repeat_time")

        self.learning_rate = self.config.getfloat("Hyper Parameters", "learning_rate")
        self.echo = self.config.getint("Hyper Parameters", "echo")
        self.type = self.config.get("Hyper Parameters", "type")
        self.log_dir = self.config.get("Hyper Parameters", "log_dir") + self.__model_name + "/log/"
        self.loss_fn = self.config.get("Hyper Parameters", "loss_fn")
        self.opt_fn = self.config.get("Hyper Parameters", "opt_fn")
        self.acc_fn = self.config.get("Hyper Parameters", "acc_fn")
        self.model_dir = self.config.get("Hyper Parameters", "model_dir") + self.__model_name + "/ckp/"

    def __init_network(self):
        self.layers = self.config.sections()
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
            output_size = self.config.getint("Output", "unit")
            self.y_ = tf.placeholder("float", [None, output_size])
            act = fn_util.get_act_fn(self.config.get("Output", "act_fn"))
            self.network = tf.contrib.layers.fully_connected(self.network[-1], output_size, activation_fn=act)
            print("Building Output Layer:Output Size =>" + str(output_size))

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
            writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            saver = tf.train.Saver(max_to_keep=5)

            next_xs, next_ys = iterator.get_next()

            init = tf.global_variables_initializer()
            sess.run(init)

            global_step = 0
            while global_step < self.echo:
                raw_xs, raw_ys = sess.run([next_xs, next_ys])
                # print(raw_xs)
                # print(raw_ys)
                batch_xs = raw_xs # No need to use dict_to_list because MNIST is ndarry
                batch_xs = batch_xs.reshape((-1, self.time_steps, self.input_size))
                batch_ys = dl_util.one_hot(raw_ys, boundaries=[0, 1, 2, 3, 4, 5, 6, 7, 8])
                #print(batch_xs)
                #print(batch_ys)

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _, cost, accuracy = sess.run([merged, self.train_step, self.cost, self.accuracy],
                                                      feed_dict={self.x: batch_xs, self.y_: batch_ys},
                                                      options=run_options,
                                                      run_metadata=run_metadata)
                writer.add_summary(summary, global_step)

                if min_cost > cost:
                    saver.save(sess, self.model_dir + self.__model_name, global_step=global_step)
                    min_cost = cost

                if best_accuracy <= accuracy:
                    best_accuracy = accuracy

                if global_step > 10000:
                    avg_accuracy = avg_accuracy + accuracy

                if global_step % 100 == 0:
                    print (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    print("Step " + str(global_step) + ": cost is " + str(cost))
                    _, acc = sess.run([merged, self.accuracy], feed_dict={self.x: batch_xs, self.y_: batch_ys})
                    print("Accuracy is: " + str(acc))

                global_step = global_step + 1
            writer.close()

            print("----------------------------------------------------------")
            print("Best Accuracy is: " + str(best_accuracy))
            print("Average Accuracy is: " + str(round(avg_accuracy / (self.echo - 10000), 2)))

    def eval(self, dataset):
        pass

    def predict(self, batch_x):
        pass


class ModelNotTrained(Exception):
    def __init__(self):
        print("Model is not trained yet")


if __name__ == "__main__":
    my_config_file = "/Users/alex/Desktop/StockLearner/config/mnist_rnn_baseline.cls"
    rnn = RNN(my_config_file, "mnist_baseline")

    from test import mnist
    train_dataset = mnist.train(mnist.MNIST_LOCAL_DIR)
    rnn.train(train_dataset)
