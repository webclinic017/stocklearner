from .base import RLBaseAgent
from rl.dqn.replay_buffer import ReplayBuffer
from util import model_util
import random


class DQNAgent(RLBaseAgent):
    def __init__(self,
                 network_config_path,
                 buffer_size=10000,
                 gamma=1.0,
                 batch_size=32,
                 target_network_update_freq=500,
                 learning_freq=200,
                 epsilon=0.9):

        super(DQNAgent, self).__init__()

        self._replay_buffer = ReplayBuffer(buffer_size)
        self._gamma = gamma
        self._batch_size = batch_size
        self._target_network_update_freq = target_network_update_freq
        self._epsilon = epsilon
        self._network_config_path = network_config_path

        self.learning_freq = learning_freq
        self.prediction_network = None
        self.target_network = None

        self.build_dqn()

    def build_dqn(self):
        # TODO
        self.prediction_network = model_util.get_model(self._network_config_path, "eval_network")
        self.target_network = model_util.get_model(self._network_config_path, "target_network")

    def choose_action(self, s_t):
        if random.random() < self._epsilon:
            action = self.random_action()
        else:
            # TODO
            # action = self.q_action.eval({self.s_t: [s_t]})[0]
            action = self.random_action()
        return action

    def store(self, s_t, action, reward, s_t_1, done):
        self._replay_buffer.add(s_t, action, reward, s_t_1, done)

    def study(self):
        # TODO
        pass
