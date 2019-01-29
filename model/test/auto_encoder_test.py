from model.auto_encoder import *

if __name__ == "__main__":
    from feed.csv_data import csv_input_fn
    config_file_path = "../config_file/mnist_auto_encoder_baseline.cls"
    training_data_path = "D:\\Output\\000017\\"
    training_dataset = csv_input_fn(training_data_path)

    ae = AutoEncoder(config_file=config_file_path)
    # ae.train(dataset=training_dataset)
