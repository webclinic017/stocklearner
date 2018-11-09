import configparser
import tensorflow as tf
from util import log_util


class AutoEncoder:
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
        self.tensorboard_summary_enabled = self.config.get("Hyper Parameters", "enable_tensorboard_log")
        
        self.logger = log_util.get_file_logger(self.model_name, self.log_dir + self.model_name + ".txt")

    def __init_network(self):
        pass
