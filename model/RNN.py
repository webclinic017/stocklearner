import configparser
import tensorflow as tf
from util import fn_util
from util import dl_util


class RNN:
    def __init__(self, config_file, model_name):
        self.__model_name = model_name
        self.__config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(self.__config_file)

        self.__init_hyper_param()
        self.__init_network()

    def __init_hyper_param(self):
        self.batch_size = self.config.getint('Dataset', 'batch_size')
        self.repeat = self.config.getint('Dataset', 'repeat_time')

        self.learning_rate = self.config.getfloat('Hyper Parameters', 'learning_rate')
        self.echo = self.config.getint('Hyper Parameters', 'echo')
        self.type = self.config.get('Hyper Parameters', 'type')
        self.log_dir = self.config.get('Hyper Parameters', 'log_dir') + self.__model_name + "/log/"
        self.loss_fn = self.config.get('Hyper Parameters', 'loss_fn')
        self.opt_fn = self.config.get('Hyper Parameters', 'opt_fn')
        self.acc_fn = self.config.get('Hyper Parameters', 'acc_fn')
        self.model_dir = self.config.get('Hyper Parameters', 'model_dir') + self.__model_name + "/ckp/"

    def __init_network(self):
        with tf.name_scope("Input"):
            input_size = self.config.getint('Input', 'unit')
            time_steps = self.config.getint("RNN Layer", "time_steps")
            self.x = tf.placeholder("float", [None, time_steps, input_size])
            self.network = tf.unstack(self.x, time_steps, 1)
            print("Building Input Layer:Input Size =>" + str(input_size))
            print("Building Input Layer:Time Steps =>" + str(time_steps))

        cell_type = self.config.get("RNN Layer", "cell_type")
        print("Building RNN Cells:Cell Type =>" + cell_type)
        with tf.name_scope(cell_type + "_layer"):
            hidden_cells = self.config.getint("RNN Layer", "hidden_cells")
            forget_bias = self.config.getfloat("RNN Layer", "forget_bias")
            act = fn_util.get_act_fn(self.config.get("RNN Layer", 'act_fn'))
            print("Building RNN Cells:Hidden Cells =>" + str(hidden_cells))

            cells = dl_util.get_rnn_cells(cell_type, hidden_cells, forget_bias)
            self.network = tf.contrib.rnn.static_rnn(cells, self.network, dtype=tf.float32)
            print("Using Static RNN")

        with tf.name_scope("Output"):
            output_size = self.config.getint('Output', 'unit')
            self.y_ = tf.placeholder("float", [None, output_size])
            self.network = tf.contrib.layers.fully_connected(self.network[-1], output_size, activation_fn=act)
            print("Building Output Layer:Output Size =>" + str(output_size))

        with tf.name_scope('Loss'):
            with tf.name_scope(self.loss_fn):
                self.cost = fn_util.get_loss_fn(self.loss_fn, self.y_, self.network)
                tf.summary.scalar(self.loss_fn, self.cost)
        with tf.name_scope('Train_Step'):
            optimizer = fn_util.get_opt_fn(self.opt_fn)
            self.train_step = optimizer(self.learning_rate).minimize(self.cost)

        with tf.name_scope('Accuracy'):
            self.accuracy = fn_util.get_acc_fn(self.acc_fn, self.y_, self.network)

    def train(self, dataset):
        pass

    def eval(self, dataset):
        pass

    def predict(self, batch_x):
        pass


if __name__ == "__main__":
    my_config_file = "/Users/alex/Desktop/StockLearner/config/mnist_rnn_baseline.cls"
    rnn = RNN(my_config_file, "mnist_baseline")