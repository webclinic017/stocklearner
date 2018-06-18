import os
import configparser
from os.path import join

CONFIG_FILE_NAME = "app.config"


class StockLearner:
    def __init__(self, app_config_path):
        self.config = configparser.ConfigParser()
        self.config.read(app_config_path)
        self.__data_feed_path = self.config.get("Application", "data_feed_path")
        self.__dnn_model = self.config.get("Application", "dnn_model")
        self.__dnn_type = self.config.get("Application", "dnn_type")
        self.__dnn_config_path = join(join(os.getcwd(), "config"), self.__dnn_model + "." + self.__dnn_type)
        print(self.__dnn_config_path)

    def learn(self):
        pass

    def test(self):
        pass

if __name__ == "__main__":
    config_file_path = join(os.getcwd(), CONFIG_FILE_NAME)
    student = StockLearner(config_file_path)
    student.learn()
