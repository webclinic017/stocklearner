from feed import csv_data as stock_data
from util import model_util
from os.path import join, isdir
from os import listdir
from util import log_util
import configparser

CONFIG_FILE_PATH = "./app.config"


logger = log_util.get_file_logger("sl_ops.py", "main.log")

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE_PATH)

    training_data_path = config.get("Data", "training_data_path")
    eval_data_path = config.get("Data", "eval_data_path")

    config_list_path = config.get("Train", "config_list_path")
    
    config_file_list = [join(config_list_path, f) for f in listdir(config_list_path) if f != ".DS_Store" and not isdir(join(config_list_path, f))]
    logger.info("Config file list path: " + config_list_path)

    train_dataset = stock_data.csv_input_fn(training_data_path)
    eval_dataset = stock_data.csv_input_fn(eval_data_path)

    # Only run the first network config for testing
    for config_file in config_file_list[:1]:
        logger.info("Current config_file file is: " + config_file)
        try:
            model = model_util.get_model(config_file)
            model.train(train_dataset)
            # model.eval(eval_dataset)
        except Exception as ex:
            logger.info(ex)
