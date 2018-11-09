# from model.MLP import MLP
from feed import csv_data as stock_data
from util import model_util
import tensorflow as tf
import configparser
import multiprocessing as mp
from os.path import join
from os import listdir
from util import log_util

CONFIG_FILE_PATH = "./app.config"
logger = log_util.get_file_logger("main.py", "main.log")


def main(argv):
    try:
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE_PATH)

        training_data_path = config.get("Data", "training_data_path")
        eval_data_path = config.get("Data", "eval_data_path")

        pool_size = config.getint("Execution", "pool_size")
        config_list_path = config.get("Execution", "config_list_path")

        config_file_list = [join(config_list_path, f) for f in listdir(config_list_path) if f != ".DS_Store"]
        logger.info("Pool size: " + str(pool_size))
        logger.info("Config file list path: " + config_list_path)

        train_dataset = stock_data.csv_input_fn(training_data_path)
        eval_dataset = stock_data.csv_input_fn(eval_data_path)

        pool = mp.Pool(processes=pool_size)
        for config_file in config_file_list:
            logger.info("Current config file is: " + config_file)
            pool.apply_async(machine_learning, (config_file, train_dataset, eval_dataset))
        pool.close()
        pool.join()
    except Exception as ex:
        print(ex)
        logger.info(ex)


def machine_learning(config_file, train_dataset, eval_dataset):
    try:
        model = model_util.get_model(config_file)
        model.train(train_dataset)
        model.eval(eval_dataset)
    except Exception as ex:
        print(ex)
        logger.info(ex)

if __name__ == "__main__":
    tf.app.run()
