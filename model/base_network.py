import tensorflow as tf
import configparser
from util import log_util
from util import fn_util
from util import dl_util
import os
import datetime


# TODO: eager execution
# tf.enable_eager_execution()


class Network:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file)

        self.__init_hyper_param()

        self.cost = None
        self.train_step = None
        self.accuracy = None

    def __init_hyper_param(self):
        self.model_name = self.config.get("Model", "name")
        self.model_type = self.config.get("Model", "type")
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
        self.tensorboard_summary_enabled = self.config.get("Hyper Parameters", "enable_tensorboard_log")

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.logger = log_util.get_file_logger(self.model_name, self.log_dir + self.model_name + ".txt")

    def _init_network(self):
        pass

    # def _var_summaries(self, var):
    #     with tf.name_scope("Summaries"):
    #         mean = tf.reduce_mean(var)
    #         tf.summary.scalar("mean", mean)
    #         with tf.name_scope("stddev"):
    #             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    #         tf.summary.scalar("stddev", stddev)
    #         tf.summary.scalar("max", tf.reduce_max(var))
    #         tf.summary.scalar("min", tf.reduce_min(var))
    #         tf.summary.histogram("histogram", var)

    def _add_train_ops(self):
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
        if self.model_type == "RNN":
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.time_steps * self.batch_size))
            # dataset = dataset.batch(self.batch_size * self.time_steps, drop_remainder= True) # available in tf 1.10
        else:
            dataset = dataset.batch(self.batch_size)

        dataset = dataset.repeat(self.repeat)
        iterator = dataset.make_one_shot_iterator()

        # 2018/08/06 - To fully utilize CPUs for training
        # config = tf.ConfigProto(device_count={"CPU": 8},
        #                         inter_op_parallelism_threads=16,
        #                         intra_op_parallelism_threads=16,
        #                         log_device_placement=True
        #                         )
        #
        # with tf.Session(config=config) as sess:
        with tf.Session() as sess:
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

            current_step = 0
            while current_step < self.echo:
                raw_xs, raw_ys = sess.run([next_xs, next_ys])

                batch_xs = dl_util.dict_to_list(raw_xs)

                if self.type == "classification":
                    batch_ys = dl_util.one_hot(raw_ys)
                else:
                    batch_ys = raw_ys

                if self.model_type == "RNN":
                    batch_xs = batch_xs.reshape((-1, self.time_steps, self.input_size))
                    batch_ys = dl_util.rnn_output_split(batch_ys, self.time_steps, self.output_size)

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _, cost, accuracy = sess.run([merged, self.train_step, self.cost, self.accuracy],
                                                      feed_dict={self.x: batch_xs, self.y_: batch_ys},
                                                      options=run_options,
                                                      run_metadata=run_metadata)

                if self.tensorboard_summary_enabled:
                    writer.add_summary(summary, current_step)

                if min_cost > cost:
                    saver.save(sess, self.model_dir + self.model_name, global_step=current_step)
                    min_cost = cost

                if best_accuracy <= accuracy:
                    best_accuracy = accuracy

                if current_step > 10000:
                    avg_accuracy = avg_accuracy + accuracy

                if current_step % 100 == 0:
                    self.logger.info("Step " + str(current_step) + ": cost is " + str(cost))
                    self.logger.info("Step " + str(current_step) + ": cost is " + str(cost))
                    _, acc = sess.run([merged, self.accuracy], feed_dict={self.x: batch_xs, self.y_: batch_ys})
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

    def eval(self, dataset):
        if not os.path.exists(self.model_dir):
            raise ModelNotTrained()

        if len(os.listdir(self.model_dir)) > 0:
            self.logger.info("----------------------------------------------------------")

            if self.model_type == "RNN":
                dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.time_steps * self.batch_size))
            else:
                dataset = dataset.batch(self.batch_size)

            dataset = dataset.repeat(self.repeat)
            iterator = dataset.make_one_shot_iterator()

            with tf.Session() as sess:
                next_xs, next_ys = iterator.get_next()

                init = tf.global_variables_initializer()
                sess.run(init)

                saver = tf.train.Saver()
                # saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(self.model_dir)+".meta")
                # self.logger.info(tf.train.latest_checkpoint(self.model_dir))
                saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))

                raw_xs, raw_ys = sess.run([next_xs, next_ys])

                batch_xs = dl_util.dict_to_list(raw_xs)
                if self.type == "classification":
                    batch_ys = dl_util.one_hot(raw_ys)
                else:
                    batch_ys = raw_ys

                if self.model_type == "RNN":
                    batch_xs = batch_xs.reshape((-1, self.time_steps, self.input_size))
                    batch_ys = dl_util.rnn_output_split(batch_ys, self.time_steps, self.output_size)

                acc = sess.run(self.accuracy, feed_dict={self.x: batch_xs, self.y_: batch_ys})
                self.logger.info("Accuracy for evaluation is: " + str(acc))
        else:
            raise ModelNotTrained()

    def predict(self, batch_x):
        pass


class ModelNotTrained(Exception):
    def __init__(self):
        self.logger.info("Model is not trained yet")
