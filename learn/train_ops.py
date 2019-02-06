import tensorflow as tf
import configparser
import datetime
from util import fn_util
from util import dl_util
from util import log_util
from feed import csv_data as stock_data


class TrainOps:
    def __init__(self, train_ops_name=""):
        self._train_ops_name = train_ops_name
        self.train_config = None
        self.dataset_config = None

        self.network = None
        self.dataset = None
        self.cost = None
        self.accuracy = None
        self.train_step = None

        self.logger = log_util.get_file_logger(train_ops_name, train_ops_name + ".log")

    def __load_dataset_config(self, dataset_config_file):
        config = configparser.ConfigParser()
        config.read(dataset_config_file)

        self.batch_size = config.getint("dataset", "batch_size")
        self.repeat = config.getint("dataset", "repeat_time")

    def __load_train_config(self, train_config_file):
        config = configparser.ConfigParser()
        config.read(train_config_file)

        self.is_training = config.getboolean("hyper parameters", "is_training")
        self.echo = config.getint("hyper parameters", "echo")
        self.type = config.get("hyper parameters", "train_type")
        self.learning_rate = config.getfloat("hyper parameters", "learning_rate")
        self.loss_fn = config.get("hyper parameters", "loss_fn")
        self.opt_fn = config.get("hyper parameters", "opt_fn")
        self.acc_fn = config.get("hyper parameters", "acc_fn")
        self.tensorboard_summary_enabled = config.get("hyper parameters", "enable_tensorboard_log")
        self.log_dir = config.get("hyper parameters", "log_dir") + self._train_ops_name + "/log/"
        self.model_dir = config.get("hyper parameters", "model_dir") + self._train_ops_name + "/ckp/"

    def add_network(self, network):
        self.network = network

    def add_dataset(self, dataset, dataset_config_file):
        self.dataset = dataset
        self.__load_dataset_config(dataset_config_file)

    def add_train_ops(self, train_config_file):
        self.__load_train_config(train_config_file)

        # TODO: add L1 & L2 normalization
        with tf.name_scope("loss"):
            with tf.name_scope(self.loss_fn):
                self.cost = fn_util.get_loss_fn(self.loss_fn, self.network.y_, self.network.y)
                tf.summary.scalar(self.loss_fn, self.cost)

        with tf.name_scope("train_step"):
            optimizer = fn_util.get_opt_fn(self.opt_fn)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = optimizer(self.learning_rate).minimize(self.cost)

        with tf.name_scope("accuracy"):
            self.accuracy = fn_util.get_acc_fn(self.acc_fn, self.network.y_, self.network.y)

    def train(self):
        if self.network is None or self.dataset is None or self.accuracy is None:
            raise ValueError

        if self.network.model_type == "RNN":
            self.dataset = self.dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.network.time_steps * self.batch_size))
            # dataset = dataset.batch(self.batch_size * self.time_steps, drop_remainder= True) # available in tf 1.10
        else:
            self.dataset = self.dataset.batch(self.batch_size)

        self.dataset = self.dataset.repeat(self.repeat)
        iterator = self.dataset.make_one_shot_iterator()

        # 2018/08/06 - To fully utilize CPUs for training
        # config_file = tf.ConfigProto(device_count={"CPU": 8},
        #                         inter_op_parallelism_threads=16,
        #                         intra_op_parallelism_threads=16,
        #                         log_device_placement=True
        #                         )
        #
        # with tf.Session(config_file=config_file) as sess:
        with tf.Session() as sess:
            min_cost = 100
            best_accuracy = 0
            avg_accuracy = 0
            merged = tf.summary.merge_all()
            saver = tf.train.Saver(max_to_keep=5)

            if self.tensorboard_summary_enabled:
                print("tensorboard log dir: " + self.log_dir)
                writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            next_xs, next_ys = iterator.get_next()

            init = tf.global_variables_initializer()
            sess.run(init)

            current_step = 0
            while current_step < self.echo:
                raw_xs, raw_ys = sess.run([next_xs, next_ys])

                batch_xs = dl_util.dict_to_list(raw_xs)

                if self.type == "classification":
                    batch_ys = dl_util.one_hot(raw_ys)
                else:
                    batch_ys = raw_ys

                if self.network.model_type == "RNN":
                    batch_xs = batch_xs.reshape((-1, self.network.time_steps, self.network.input_size))
                    batch_ys = dl_util.rnn_output_split(batch_ys, self.network.time_steps, self.network.output_size)

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _, cost, accuracy = sess.run([merged, self.train_step, self.cost, self.accuracy],
                                                      feed_dict={self.network.x: batch_xs,
                                                                 self.network.y_: batch_ys,
                                                                 self.network.is_training:self.is_training},
                                                      options=run_options,
                                                      run_metadata=run_metadata)

                if self.tensorboard_summary_enabled:
                    writer.add_summary(summary, current_step)

                if min_cost > cost:
                    # print("checkpoint save dir: " + self.model_dir + self._train_ops_name)
                    saver.save(sess, self.model_dir + self._train_ops_name, global_step=current_step)
                    min_cost = cost

                if best_accuracy <= accuracy:
                    best_accuracy = accuracy

                if current_step > 10000:
                    avg_accuracy = avg_accuracy + accuracy

                if current_step % 100 == 0:
                    self.logger.info("Step " + str(current_step) + ": cost is " + str(cost))
                    self.logger.info("Step " + str(current_step) + ": cost is " + str(cost))
                    _, acc = sess.run([merged, self.accuracy],
                                      feed_dict={self.network.x: batch_xs,
                                                 self.network.y_: batch_ys,
                                                 self.network.is_training: self.is_training})
                    self.logger.info("Accuracy is: " + str(acc))
                    self.logger.info("Accuracy is: " + str(acc))

                current_step = current_step + 1

            if self.tensorboard_summary_enabled:
                writer.close()

            self.logger.info("----------------------------------------------------------")
            self.logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            self.logger.info("Best Accuracy is: " + str(best_accuracy))
            self.logger.info("Best Accuracy is: " + str(best_accuracy))
            self.logger.info("Average Accuracy is: " + str(round(avg_accuracy / (self.echo - 10000), 2)))
            self.logger.info("Average Accuracy is: " + str(round(avg_accuracy / (self.echo - 10000), 2)))

            # if self.type == "classification":
            #     f1_score = tf.contrib.metrics.f1_score(labels=batch_ys, predictions=self.y_)
            #     sess.run(f1_score)

    def eval(self, batch_x):
        raise NotImplementedError
