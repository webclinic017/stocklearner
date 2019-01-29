import configparser


class BaseConfig(object):
    def __init__(self, config_file_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_file_path)


class NetworkConfig(BaseConfig):
    def __init__(self, config_file):
        NetworkConfig.__init__(self, config_file)
        self.model_name = self.config.get("Model", "name")
        self.model_type = self.config.get("Model", "type")


class TrainAppConfig(BaseConfig):
    def __init__(self, config_file):
        TrainAppConfig.__init__(self, config_file)
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

# class RLAppConfig(BaseConfig):
#     def __init__(self, config_file):
#         RLAppConfig.__init__(self, config_file)
#
#
# class MLPConfig(NetworkConfig):
#     def __init__(self, config_file):
#         MLPConfig.__init__(self, config_file)
#
#
# class RNNConfig(NetworkConfig):
#     def __init__(self, config_file):
#         RNNConfig.__init__(self, config_file)


class AutoEncoderConfig(NetworkConfig):
    def __init__(self, config_file):
        AutoEncoderConfig.__init__(self, config_file)


class CNNConfig(NetworkConfig):
    def __init__(self, config_file):
        CNNConfig.__init__(self, config_file)