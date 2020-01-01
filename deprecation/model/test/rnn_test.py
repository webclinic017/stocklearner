from deprecation.model.rnn import *

if __name__ == "__main__":
    my_config_file = "../config_file/stock_rnn_baseline.cls"
    rnn = RNN(my_config_file)

    from feed.csv_data import csv_input_fn
    config_file_path = "../config_file/stock_mlp_baseline.cls"
    training_data_path = "D:\\Output\\000017\\"
    training_dataset = csv_input_fn(training_data_path)

    rnn.train(training_dataset)
