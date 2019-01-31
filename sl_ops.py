from feed import csv_data as stock_data
from learn.train_ops import TrainOps
from util import model_util
from util import log_util
import configparser

CONFIG_FILE_PATH = "./app.config"

logger = log_util.get_file_logger("sl_ops.py", "sl_ops.log")

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE_PATH)

    train_ops_name = config.get("train_ops", "name")
    training_data_path = config.get("data_path", "training_data_path")
    eval_data_path = config.get("data_path", "eval_data_path")

    network_config_file = config.get("config_file", "network_config_file")
    dataset_config_file = config.get("config_file", "dataset_config_file")
    train_ops_config_file = config.get("config_file", "train_ops_config_file")

    train_dataset = stock_data.csv_input_fn(training_data_path)
    eval_dataset = stock_data.csv_input_fn(eval_data_path)

    try:
        network = model_util.get_network(network_config_file, "testing")
        train_ops = TrainOps(train_ops_name)
        train_ops.add_network(network)
        train_ops.add_dataset(train_dataset, dataset_config_file)
        train_ops.add_train_ops(train_ops_config_file)
        train_ops.train()

        # train_ops.eval(eval_dataset)
    except Exception as ex:
        logger.info(ex)
