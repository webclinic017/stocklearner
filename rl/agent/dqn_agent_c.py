from .base import RLBaseAgent
from rl.dqn.replay_buffer import ReplayBuffer
from rl.ops import linear, clipped_error
import tensorflow as tf
import numpy as np
import os
import random

# TODO: 1. configure parameters
#       2. Double Q
#       3. Duel DQN
#       4. Prioritized Replay Buffer
#       3. Summary
#       4. Checkpoint - DONE
#       5. Copy network vars from target to prediction
#       6. learning rate ops


class DQNConfig:
    def __init__(self, **kwargs):
        self.buffer_size = 1000

        self.input_size = 8
        self.l1_unit = 512
        self.l2_unit = 256
        self.l3_unit = 128
        self.l4_unit = 64

        self.action_size = 4

        self.learning_rate = 0.00025
        self.learning_rate_step = 100
        self.learning_rate_decay_step = 5 * 10000
        self.learning_rate_decay = 0.96
        self.learning_rate_minimum = 0.00025

        self.checkpoint_dir = "chk"
        self.max_to_save = 5
        self.save_frequency = 1000

        self.learning_freq = 100
        self.batch_size = 32
        self.target_q_update_step = 50
        self.discount = 0.99
        self.train_frequency = 4
        self.learn_start = 32
        self.double_q = False

        self.summary_log_dir = "summary"
        self.enable_summary = False


class DQNAgent(RLBaseAgent):
    def __init__(self, config, sess):
        super(DQNAgent, self).__init__()

        self.sess = sess
        self.config = config
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        self.saver = None

        with tf.variable_scope("step"):
            self.step_op = tf.Variable(0, trainable=False, name="step")
            self.step_input = tf.placeholder(dtype=tf.int32, shape=None, name="step_input")
            self.step_assign_op = self.step_op.assign(self.step_input)

        self.w = {}
        self.t_w = {}

        self.build_dqn()

        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_save)

        if self.config.enable_summary:
            self.writer = tf.summary.FileWriter(self.config.summary_log_dir, sess.graph)
        else:
            self.writer = None

    def build_dqn(self):
        activation_fn = tf.nn.relu6

        with tf.variable_scope("prediction"):
            self.s_t = tf.placeholder(dtype=tf.float32, shape=[None, self.config.input_size], name="s_t")
            self.l1, self.w["l1_w"], self.w["l1_b"] = linear(self.s_t, self.config.l1_unit, activation_fn, name="l1")
            self.l2, self.w["l2_w"], self.w["l2_b"] = linear(self.l1, self.config.l2_unit, activation_fn, name="l2")
            self.l3, self.w["l3_w"], self.w["l3_b"] = linear(self.l2, self.config.l3_unit, activation_fn, name="l3")
            self.l4, self.w["l4_w"], self.w["l4_b"] = linear(self.l3, self.config.l4_unit, activation_fn, name="l4")
            self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.config.action_size, name='q')

        self.q_action = tf.argmax(self.q, axis=1)

        with tf.variable_scope("target"):
            self.target_s_t = tf.placeholder(dtype=tf.float32, shape=[None, self.config.input_size], name="s_t")
            self.target_l1, self.t_w["l1_w"], self.t_w["l1_b"] = linear(self.target_s_t, self.config.l1_unit, activation_fn, name="l1")
            self.target_l2, self.t_w["l2_w"], self.t_w["l2_b"] = linear(self.target_l1, self.config.l2_unit, activation_fn, name="l2")
            self.target_l3, self.t_w["l3_w"], self.t_w["l3_b"] = linear(self.target_l2, self.config.l3_unit, activation_fn, name="l3")
            self.target_l4, self.t_w["l4_w"], self.t_w["l4_b"] = linear(self.target_l3, self.config.l4_unit, activation_fn, name="l4")
            self.target_q, self.t_w['q_w'], self.t_w['q_b'] = linear(self.target_l4, self.config.action_size, name='target_q')

        self.target_q_idx = tf.placeholder(dtype=tf.int32, shape=[None, None], name='outputs_idx')
        self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.w.keys():
                # prediction -> l1_w,l1_b, l2_w, l2_b
                self.t_w_input[name] = tf.placeholder(dtype=tf.float32, shape=self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

        # optimizer
        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder(dtype=tf.float32, shape=[None], name='target_q_t')
            self.action = tf.placeholder(dtype=tf.int32, shape=[None], name='action')

            action_one_hot = tf.one_hot(self.action, self.config.action_size, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

            self.delta = self.target_q_t - q_acted

            self.global_step = tf.Variable(0, trainable=False)

            self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
            self.learning_rate_step = tf.placeholder(dtype=tf.int32, shape=None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.config.learning_rate_minimum,
                                               tf.train.exponential_decay(
                                                   self.config.learning_rate,
                                                   self.config.learning_rate_step,
                                                   self.config.learning_rate_decay_step,
                                                   self.config.learning_rate_decay,
                                                   staircase=True))
            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)
            # self.optimizer = tf.train.AdagradOptimizer(self.config.learning_rate).minimize(self.loss)

        # tf.initialize_all_variables().run()
        tf.global_variables_initializer().run()
        # self.saver = tf.train.Saver({"w": self.w.values(), "step_op": [self.step_op]}, max_to_keep=30)
        self.load_model()
        self.update_target_q_network()

    def update_target_q_network(self):
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

    def save_model(self, step):
        print(" [*] Saving checkpoints...")
        # model_name = type(self).__name__

        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
        self.saver.save(self.sess, self.config.checkpoint_dir, global_step=step)

    def load_model(self):
        print(" [*] Loading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(self.config.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.config.checkpoint_dir, ckpt_name)
            self.saver.restore(self.sess, fname)
            print(" [*] Load SUCCESS: %s" % fname)
            return True
        else:
            print(" [!] Load FAILED: %s" % self.config.checkpoint_dir)
            return False

    def study(self, global_step):
        if global_step > self.config.learn_start:
            if global_step % self.config.train_frequency == 0:
                self.q_learning_mini_batch(global_step)

            if global_step % self.config.target_q_update_step == 0:
                self.update_target_q_network()

            if global_step % self.config.save_frequency == 0:
                self.save_model(global_step)

    def q_learning_mini_batch(self, step):
        if len(self.replay_buffer) < self.config.batch_size:
            return

        s_t, action, reward, s_t_plus_1, terminal = self.replay_buffer.sample(self.config.batch_size)

        if self.config.double_q:
            # Double Q-learning
            pred_action = self.q_action.eval({self.s_t: s_t_plus_1})
            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
            self.target_s_t: s_t_plus_1,
            self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]})
            target_q_t = (1. - terminal) * self.config.discount * q_t_plus_1_with_pred_action + reward
        else:
            q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})
            terminal = np.array(terminal) + 0.
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
            target_q_t = (1. - terminal) * self.config.discount * max_q_t_plus_1 + reward

        _, q_t, loss = self.sess.run([self.optimizer, self.q, self.loss], {
            self.target_q_t: target_q_t,
            self.action: action,
            self.s_t: s_t,
            self.learning_rate_step: step,
            })

        # self.writer.add_summary(summary_str, self.step)
        # self.total_loss += loss
        # self.total_q += q_t.mean()
        # self.update_count += 1

    def store(self, observe, reward, action, next_observe, terminal):
        if observe is None or next_observe is None:
            return
        self.replay_buffer.add(observe, reward, action, next_observe, terminal)

    def choose_action(self, s_t, test_ep=None):
        # ep = test_ep or (self.ep_end +
        #                  max(0., (self.ep_start - self.ep_end)
        #                      * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))
        ep = 0.9
        if random.random() < ep:
            action = random.randrange(self.config.action_size)
        else:
            action = self.q_action.eval({self.s_t: [s_t]})[0]
        return action
