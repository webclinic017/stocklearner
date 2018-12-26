from model.mlp import *
from model.rnn import *

logger = log_util.get_file_logger("model_util.py", "main.log")


def get_model(model_config_path, network_name=""):
    config = configparser.ConfigParser()
    config.read(model_config_path)
    model_type = config.get("Model", "type")
    logger.info("Current model type is " + model_type)

    if model_type == "MLP":
        model = MLP(config_file=model_config_path, network_name=network_name)
        return model

    if model_type == "RNN":
        model = RNN(config_file=model_config_path)
        return model
    raise ModelTypeNotFound()


class ModelTypeNotFound(Exception):
    def __init__(self):
        logger.error("Model Type is not found")