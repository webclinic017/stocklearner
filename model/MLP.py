import os
import datetime
import configparser
import tensorflow as tf
# import traceback
# import hashlib
# import numpy as np
from util import fn_util
from util import dl_util

IRIS_BOUNDARIES = [0, 1]


class MLP:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file)

        self.__init_hyper_param()
        self.__init_network()

    def __init_hyper_param(self):
        self.model_name = self.config.get("Model", "name")
        self.batch_size = self.config.getint("Dataset", "batch_size")
        self.repeat = self.config.getint("Dataset", "repeat_time")

        self.learning_rate = self.config.getfloat("Hyper Parameters", "learning_rate")
        self.echo = self.config.getint("Hyper Parameters", "echo")
        self.type = self.config.get("Hyper Parameters", "type")
        self.log_dir = self.config.get("Hyper Parameters", "log_dir") + self.model_name + "/log/"
        self.loss_fn = self.config.get("Hyper Parameters", "loss_fn")
        self.opt_fn = self.config.get("Hyper Parameters", "opt_fn")
        self.acc_fn = self.config.get("Hyper Parameters", "acc_fn")
        self.model_dir = self.config.get("Hyper Parameters", "model_dir") + self.model_name + "/ckp/"

    @staticmethod
    def __var_summaries(var):
        with tf.name_scope("Summaries"):
            mean = tf.reduce_mean(var)
            tf.summary.scalar("mean", mean)
            with tf.name_scope("stddev"):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar("stddev", stddev)
            tf.summary.scalar("max", tf.reduce_max(var))
            tf.summary.scalar("min", tf.reduce_min(var))
            tf.summary.histogram("histogram", var)

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
                        self.__var_summaries(W)
                        b = tf.get_variable("Bias",
                                            dtype=tf.float32,
                                            initializer=tf.constant_initializer(value=0.1),
                                            shape=[n_units])
                        self.__var_summaries(b)
                    with tf.name_scope("Wx_plus_b"):
                        preactivate = tf.matmul(self.network, W) + b
                        tf.summary.histogram("pre_activation", preactivate)
                    if layer == "Output":
                        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, n_units], name=layer)
                        self.network = preactivate
                        print("Building Output layer:Output Size =>[" + str(n_in) + ", " + str(n_units) + "]")
                    else:
                        act = fn_util.get_act_fn(self.config.get(layer, "act_fn"))
                        self.network = act(preactivate)
                        tf.summary.histogram("activation", self.network)
                        print("Building hidden layers:Name =>" + layer + " Size =>[" + str(n_in) + ", " + str(n_units) + "]")
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
            writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            saver = tf.train.Saver(max_to_keep=5)

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
                writer.add_summary(summary, global_step)

                if min_cost > cost:
                    saver.save(sess, self.model_dir + self.model_name, global_step=global_step)
                    min_cost = cost

                if best_accuracy <= accuracy:
                    best_accuracy = accuracy

                if global_step > 10000:
                    avg_accuracy = avg_accuracy + accuracy

                if global_step % 100 == 0:
                    print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    print("Step " + str(global_step) + ": cost is " + str(cost))
                    _, acc = sess.run([merged, self.accuracy], feed_dict={self.x: batch_xs, self.y_: batch_ys})
                    print("Accuracy is: " + str(acc))

                global_step = global_step + 1
            writer.close()

            print("----------------------------------------------------------")
            print("Best Accuracy is: " + str(best_accuracy))
            print("Average Accuracy is: " + str(round(avg_accuracy / (self.echo - 10000), 2)))

        # for f in [os.path.join(self.model_dir, i) for i in os.listdir(self.model_dir)]:
        #     print(f + " " + md5(f))

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

            # for f in [os.path.join(self.model_dir, i) for i in os.listdir(self.model_dir)]:
            #     print(f + " " + md5(f))
        else:
            raise ModelNotTrained()

    def predict(self, batch_x):
        if not os.path.exists(self.model_dir):
            raise ModelNotTrained()

        if len(os.listdir(self.model_dir)) > 0:
            print("----------------------------------------------------------")

            with tf.Session() as sess:
                init = tf.global_variables_initializer()
                sess.run(init)

                saver = tf.train.Saver()
                # saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(self.model_dir) + ".meta")
                saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))

                y = sess.run(self.network, feed_dict={self.x: batch_x})
                print(y)
                print(sess.run(tf.argmax(y, 1)))

            # for f in [os.path.join(self.model_dir, i) for i in os.listdir(self.model_dir)]:
            #     print(f + " " + md5(f))


class ModelNotTrained(Exception):
    def __init__(self):
        print("Model is not trained yet")


# def md5(filename):
#     hash_md5 = hashlib.md5()
#     with open(filename, "rb") as f:
#         for chunk in iter(lambda : f.read(4096), b""):
#             hash_md5.update(chunk)
#     return hash_md5.hexdigest()

# if __name__ == "__main__":
#     from test import iris
#
#     (train_x, train_y), (test_x, test_y) = iris.load_data()
#     dataset_train = iris.train_input_fn(train_x, train_y)
#     dataset_eval = iris.eval_input_fn(test_x, test_y)
#
#     my_config_file = "/Users/alex/Desktop/StockLearner/config/iris_mlp_baseline.cls"
#     mlp = MLP(config_file=my_config_file, model_name="iris_baseline")
#     mlp.train_by_dataset(dataset_train)
#     mlp.eval_by_dataset(dataset_eval)
#
#     import numpy as np
#     predict_data = np.array([
#         [5.9,3.0,4.2,1.5]  # 1
#         ,[6.9,3.1,5.4,2.1] # 2
#         ,[5.1,3.3,1.7,0.5] # 0
#         ,[6.0,3.4,4.5,1.6] # 1
#         ,[5.5,2.5,4.0,1.3] # 1
#         ,[6.2,2.9,4.3,1.3] # 1
#     ])
#     mlp.predict(predict_data)
