from .base import RLBaseAgent
from rl.dqn.replay_buffer import ReplayBuffer
import tensorflow as tf

class DQNAgent(RLBaseAgent):
    def __init__(self, config, sess):
        super(DQNAgent, self).__init__()

        self.config = config
        self._replay_buffer = ReplayBuffer(self.config.buffer_size)

        with tf.variable_scope("step"):
            self.step_op = tf.Variable(0, trainable=False, name="step")
            self.step_input = tf.placeholder(dtype=tf.int32, shape=None, name="step_inplut")
            self.step_assign_op = self.step_op.assign(self.step_input)

        self.build_dqn()

    def build_dqn(self):
        self.w = {}
        self.t_w = {}

        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu6

        with tf.variable_scope("prediction"):
            self.s_t = tf.placeholder(dtype=tf.float32, shape=[None, self.config.input_size], name="s_t")
            self.l1 = tf.layers.dense(inputs=self.s_t,
                                      units=128,
                                      kernel_initializer=initializer,
                                      bias_initializer=initializer,
                                      name="l1")
            self.l2 = tf.layers.dense(inputs=self.l1,
                                      units=64,
                                      kernel_initializer=initializer,
                                      bias_initializer=initializer,
                                      name="l2")