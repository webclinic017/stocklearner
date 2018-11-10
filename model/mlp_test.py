import hashlib
from model.mlp import *


def md5(filename):
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda : f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

if __name__ == "__main__":
    from test import iris

    # (train_x, train_y), (test_x, test_y) = iris.load_data()
    # dataset_train = iris.train_input_fn(train_x, train_y)
    # dataset_eval = iris.eval_input_fn(test_x, test_y)
    #
    # my_config_file = "/Users/alex/Desktop/StockLearner/config/iris_mlp_baseline.cls"
    # mlp = MLP(config_file=my_config_file, model_name="iris_baseline")
    # mlp.train_by_dataset(dataset_train)
    # mlp.eval_by_dataset(dataset_eval)
    #
    # import numpy as np
    # predict_data = np.array([
    #     [5.9, 3.0, 4.2, 1.5],# 1
    #     [6.9, 3.1, 5.4, 2.1],# 2
    #     [5.1, 3.3, 1.7, 0.5],# 0
    #     [6.0, 3.4, 4.5, 1.6],# 1
    #     [5.5, 2.5, 4.0, 1.3],# 1
    #     [6.2, 2.9, 4.3, 1.3]# 1
    # ])
    # mlp.predict(predict_data)

    from feed.csv_data import csv_input_fn
    config_file_path = "../config/stock_mlp_baseline.cls"
    training_data_path = "D:\\Output\\000017\\"
    training_dataset = csv_input_fn(training_data_path)

    mlp = MLP(config_file=config_file_path)
    mlp.train(dataset=training_dataset)
