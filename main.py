# from model.MLP import MLP
from feed import csv_data as stock
from util import model_util
import tensorflow as tf
import configparser

CONFIG_FILE_PATH = "./app.config"


def main(argv):
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE_PATH)

    training_data_path = config.get("Data", "training_data_path")
    eval_data_path = config.get("Data", "eval_data_path")

    model_type = config.get("Model", "model_type")
    model_name = config.get("Model", "model_name")
    model_config_path = config.get("Model", "model_config_file")

    dataset_train = stock.csv_input_fn(training_data_path)
    dataset_eval = stock.csv_input_fn(eval_data_path)

    try:
        model = model_util.get_model(model_type, model_config_path, model_name)
        model.train(dataset_train)
        model.eval(dataset_eval)
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    tf.app.run()
