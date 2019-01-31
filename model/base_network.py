import tensorflow as tf
import configparser
from util import log_util


class Network:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file)
        self.model_type = self.config.get("model", "type")

        self.logger = log_util.get_file_logger("network.py", "network.log")

    def __init_network(self):
        pass

    @staticmethod
    def _var_summaries(var):
        with tf.name_scope("Summaries"):
            mean = tf.reduce_mean(var)
            tf.summary.scalar("mean", mean)
            with tf.name_scope("stddev"):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar("stddev", stddev)
            tf.summary.scalar("max", tf.reduce_max(var))
            tf.summary.scalar("min", tf.reduce_min(var))
            tf.summary.histogram("histogram", var)
