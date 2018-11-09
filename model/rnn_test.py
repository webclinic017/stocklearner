from model.rnn import *

if __name__ == "__main__":
    my_config_file = "/Users/alex/Desktop/StockLearner/config/mnist_rnn_baseline.cls"
    rnn = RNN(my_config_file, "mnist_baseline")

    from test import mnist
    train_dataset = mnist.train(mnist.MNIST_LOCAL_DIR)
    rnn.train(train_dataset)
    pass
