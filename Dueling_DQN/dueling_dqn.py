import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

# define env
Env_Name = 'Pendulum-v0'
env = gym.make(Env_Name)
env.seed(1)

# define hyper-parameters
Log_dir = './log'
Memory_Size = 3000
L_R = 0.001
Gamma = 0.9
Ep_greedy = 0.9
Ep_greedy_increment = 0.001
Replace_target_iter = 400
Batch_size = 32
Action_space = 25
State_space = env.observation_space.shape[0]
Output_graph = False
Dueling = True
Render = False

class DuelingDQN(object):
    def __init__(self, sess):
        self.sess = sess
        self.epsilon_max = Ep_greedy
        self.epsilon = 0 if Ep_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory = np.zeros((Memory_Size, State_space * 2 + 2))

        if Output_graph:
            tf.summary.FileWriter(logdir=Log_dir, graph=self.sess.graph)

        self.cost_his = []
        self._build_net()

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [State_space, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
            if Dueling:
                with tf.variable_scope('Value'):
                    w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.matmul(l1, w2) + b2
                with tf.variable_scope('Advantage'):
                    w2 = tf.get_variable('w2', [n_l1, Action_space], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, Action_space], initializer=b_initializer, collections=c_names)
                    self.A = tf.matmul(l1, w2) + b2
                with tf.variable_scope('Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))
            else:
                with tf.variable_scope('Q'):
                    w2 = tf.get_variable('w2', [n_l1, Action_space], initializer=w_initializer, collections=c_names)
                    b2 = tf.get_variable('b2', [1, Action_space], initializer=b_initializer, collections=c_names)
                    out = tf.matmul(l1, w2) + b2
            return out
        # build eval_net
        self.s = tf.placeholder(tf.float32, [None, State_space], name='s')
        self.target_q = tf.placeholder(tf.float32, [None, Action_space], name='target_q')
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
            ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
            tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_eval, self.target_q))
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(L_R).minimize(self.loss)
        # build target_net
        self.s_ = tf.placeholder(tf.float32, [None, State_space], name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % Memory_Size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action_index = np.argmax(action_value)
        else:
            action_index = np.random.randint(0, Action_space)
        return action_index

    def replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

    def learn(self):
        if self.learn_step_counter % Replace_target_iter == 0:
            self.replace_target_params()
            print('\ntarget_params_replaced\n')
        sample_index = np.random.choice(Memory_Size, size=Batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval_next, = self.sess.run([self.q_next, self.q_eval],
                                            feed_dict={self.s_: batch_memory[:, -State_space:],
                                                       self.s: batch_memory[:, -State_space:]})
        q_eval = self.sess.run(self.q_eval, feed_dict={self.s: batch_memory[:, :State_space]})
        q_target = q_eval.copy()

        batch_index = np.arange(Batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, State_space].astype(int)
        reward = batch_memory[:, State_space + 1]
        q_target[batch_index, eval_act_index] = reward + Gamma * np.max(q_next, axis=1)
        _, self.cost = self.sess.run([self.train_op, self.loss], feed_dict={self.s: batch_memory[:, :State_space],
                                                                            self.target_q: q_target})
        self.cost_his.append(self.cost)
        self.epsilon = self.epsilon + Ep_greedy_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

if __name__ == "__main__":
    sess = tf.Session()

    DuelingDQN = DuelingDQN(sess)
    sess.run(tf.global_variables_initializer())
    total_step = 0
    for i in range(100):
        s = env.reset()
        ep_reward = 0
        for j in range(500):
            if Render:
                env.render()
            a = DuelingDQN.choose_action(observation=s)
            f_action = (a - (Action_space - 1) / 2) / ((Action_space - 1) / 4)
            s_, r, done, info = env.step([a])
            DuelingDQN.store_transition(s, a, r/10, s_)
            if total_step > Memory_Size:
                DuelingDQN.learn()
            s = s_
            total_step += 1
            ep_reward += r
            if j == 499:
                print('Episode:', i, ' Total Rewards: %i' % int(ep_reward))
                if ep_reward > -1000:
                    Render = True
                break