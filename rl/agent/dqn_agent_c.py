from .base import RLBaseAgent
from rl.dqn.replay_buffer import ReplayBuffer
from rl.ops import linear, clipped_error
import tensorflow as tf
import numpy as np
import os, random


class DQNAgent(RLBaseAgent):
    def __init__(self, config, sess):
        super(DQNAgent, self).__init__()

        self.config = config
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)
        self.saver = None

        with tf.variable_scope("step"):
            self.step_op = tf.Variable(0, trainable=False, name="step")
            self.step_input = tf.placeholder(dtype=tf.int32, shape=None, name="step_inplut")
            self.step_assign_op = self.step_op.assign(self.step_input)

        self.w = {}
        self.t_w = {}

        self.build_dqn()

    def build_dqn(self):
        activation_fn = tf.nn.relu6

        with tf.variable_scope("prediction"):
            self.s_t = tf.placeholder(dtype=tf.float32, shape=[None, self.config.input_size], name="s_t")
            self.l1, self.w["l1_w"], self.w["l1_w"] = linear(self.s_t, self.config.l1_unit, activation_fn, name="l1")
            self.l2, self.w["l2_w"], self.w["l2_w"] = linear(self.l1, self.config.l2_unit, activation_fn, name="l2")
            self.l3, self.w["l3_w"], self.w["l3_w"] = linear(self.l2, self.config.l3_unit, activation_fn, name="l3")
            self.l4, self.w["l4_w"], self.w["l4_w"] = linear(self.l3, self.config.l4_unit, activation_fn, name="l4")
            self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.config.action_space, name='q')

        self.q_action = tf.argmax(self.q, dimension=1)

        with tf.variable_scope("target"):
            self.target_s_t = tf.placeholder(dtype=tf.float32, shape=[None, self.config.input_size], name="s_t")
            self.target_l1, self.w["l1_w"], self.w["l1_w"] = linear(self.target_s_t, self.config.l1_unit, activation_fn, name="l1")
            self.target_l2, self.w["l2_w"], self.w["l2_w"] = linear(self.target_l1, self.config.l2_unit, activation_fn, name="l2")
            self.target_l3, self.w["l3_w"], self.w["l3_w"] = linear(self.target_l2, self.config.l3_unit, activation_fn, name="l3")
            self.target_l4, self.w["l4_w"], self.w["l4_w"] = linear(self.target_l3, self.config.l4_unit, activation_fn, name="l4")
            self.target_q, self.w['q_w'], self.w['q_b'] = linear(self.target_l4, self.config.action_size, name='target_q')

        self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
        self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

        # optimizer
        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
            self.action = tf.placeholder('int64', [None], name='action')

            action_one_hot = tf.one_hot(self.action, self.config.action_size, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

            self.delta = self.target_q_t - q_acted

            self.global_step = tf.Variable(0, trainable=False)

            self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                               tf.train.exponential_decay(
                                                   self.config.learning_rate,
                                                   self.config.learning_rate_step,
                                                   self.config.learning_rate_decay_step,
                                                   self.config.learning_rate_decay,
                                                   staircase=True))
            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

        tf.initialize_all_variables().run()
        self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)
        self.load_model()
        self.update_target_q_network()

    def update_target_q_network(self):
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

    def save_model(self, step=None):
        print(" [*] Saving checkpoints...")
        model_name = type(self).__name__

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

    def study(self):
        pass

    def q_learning_mini_batch(self):
        s_t, action, reward, s_t_plus_1, terminal = self.replay_buffer.sample()

        # t = time.time()
        if self.double_q:
            # Double Q-learning
            pred_action = self.q_action.eval({self.s_t: s_t_plus_1})
            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({
            self.target_s_t: s_t_plus_1,
            self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]})
            target_q_t = (1. - terminal) * self.discount * q_t_plus_1_with_pred_action + reward
        else:
            q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})
            terminal = np.array(terminal) + 0.
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
            target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

            _, q_t, loss, summary_str = self.sess.run([self.optim, self.q, self.loss, self.q_summary], {
            self.target_q_t: target_q_t,
            self.action: action,
            self.s_t: s_t,
            self.learning_rate_step: self.step,
            })

        # self.writer.add_summary(summary_str, self.step)
        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1

    def store(self, observe, reward, action, terminal):
        reward = max(self.min_reward, min(self.max_reward, reward))

        # self.history.add(observe)
        self.replay_buffer.add(observe, reward, action, terminal)

        if self.step > self.learn_start:
            if self.step % self.train_frequency == 0:
                self.q_learning_mini_batch()

            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                self.update_target_q_network()

    def choose_action(self, s_t, test_ep=None):
        ep = test_ep or (self.ep_end +
                         max(0., (self.ep_start - self.ep_end)
                             * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))
        if random.random() < ep:
            action = random.randrange(self.env.action_size)
        else:
            action = self.q_action.eval({self.s_t: [s_t]})[0]
        return action
