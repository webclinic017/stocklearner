from model.base_network import *


class AutoEncoder(Network):
    def __init__(self, config_file):
        Network.__init__(self, config_file)

        # self.__init_hyper_param()
        self.__init_network()

    def __init_hyper_param(self):
        pass

    def __init_network(self):
        pass
