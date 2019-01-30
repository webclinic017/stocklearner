import tensorflow as tf
from util import fn_util


class TrainOps:
    def __init__(self, config):
        self.config = config
        self.x = None
        self.y_ = None
        self.network = None
        self.cost = None
        self.accuracy = None
        self.train_step = None
        raise NotImplementedError

    def add_network(self, model):
        self.x = model.x
        self.y_ = model.y_
        self.network = model.network

    def add_train_ops(self):
        # TODO: add L1 & L2 normalization
        with tf.name_scope("Loss"):
            with tf.name_scope(self.loss_fn):
                self.cost = fn_util.get_loss_fn(self.loss_fn, self.y_, self.network)
                tf.summary.scalar(self.loss_fn, self.cost)

        with tf.name_scope("Train_Step"):
            optimizer = fn_util.get_opt_fn(self.opt_fn)
            self.train_step = optimizer(self.learning_rate).minimize(self.cost)

        with tf.name_scope("Accuracy"):
            self.accuracy = fn_util.get_acc_fn(self.acc_fn, self.y_, self.network)

    def train(self, dataset):
        raise NotImplementedError

    def eval(self, batch_x):
        raise NotImplementedError
