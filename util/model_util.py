from deprecation.model.rnn import *

logger = log_util.get_file_logger("model_util.py", "sl_ops.log")


def get_network(network_config_path, network_name=""):
    config = configparser.ConfigParser()
    config.read(network_config_path)
    network_type = config.get("network", "type")
    logger.info("Current model type is " + network_type)

    if network_type == "MLP":
        network = MLP(config_file=network_config_path, network_name=network_name)
        return network

    if network_type == "RNN":
        network = RNN(config_file=network_config_path, network_name=network_name)
        return network
    raise NetworkTypeNotFound()


class NetworkTypeNotFound(Exception):
    def __init__(self):
        logger.error("Network Type is not found")
