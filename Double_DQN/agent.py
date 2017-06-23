import tensorflow as tf
import numpy as np

np.random.seed(1)
tf.set_random_seed(1)

class DoubleDQN:
    def __init__(self, n_actions, n_features, learning_rate=0.005, reward_decay=0.9, e_greedy=0.9,
                 replace_target_freq=200, memory_size=3000, batch_size=32, e_greedy_increment=None, out_graph=False,
                 double_q=True, sess=None,):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_freq = replace_target_freq
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.double_q = double_q
        # total learning step
        self.learn_step_counter = 0

        # initialize memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features*2 + 2))

        # build net (eval_net and target_net)
        self.build_net()
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if out_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []

    def build_net(self):
        def build_layers(s, c_names, layer1_nums, w_initializer, b_initializer):
            with tf.variable_scope('layer1'):
                w1 = tf.get_variable('w1', [self.n_features, layer1_nums],
                                     initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, layer1_nums],
                                     initializer=b_initializer, collections=c_names)
                layer1_out = tf.nn.relu(tf.matmul(s, w1) + b1)
            with tf.variable_scope('layer2'):
                w2 = tf.get_variable('w2', [layer1_nums, self.n_actions],
                                     initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions],
                                     initializer=b_initializer, collections=c_names)
                out = tf.matmul(layer1_out, w2) + b2
            return out
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='q_target')

        with tf.variable_scope('eval_net'):
            c_names, layer1_num, w_initializer, b_initializer = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
            20, tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
            self.q_eval = build_layers(self.s, c_names, layer1_num, w_initializer, b_initializer)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self.train_step = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, layer1_num, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)
        if not hasattr(self, 'q'):
            self.q = []
            self.q_running = 0
        self.q_running = self.q_running*0.99 + 0.01*np.max(actions_value)
        self.q.append(self.q_running)
        if np.random.uniform() > self.epsilon:
            action = np.random.randint(0, self.n_actions)
        return action

    def replace_target_weight(self):
        target_weights = tf.get_collection('target_net_params')
        eval_weights = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(target_weights, eval_weights)])

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_freq == 0:
            self.replace_target_weight()
            print("\ntargetNet's weights have been replaced\n")
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            batch_indices = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            batch_indices = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[batch_indices, :]

        q_next, q_eval_next = self.sess.run([self.q_next, self.q_eval],
                                            feed_dict={self.s_: batch_memory[:, -self.n_features:],
                                                       self.s: batch_memory[:, -self.n_features:]})
        q_eval = self.sess.run(self.q_eval, feed_dict={self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_next_act = np.argmax(q_eval_next, axis=1)
            selected_q_next = q_next[batch_index, max_next_act]
        else:
            selected_q_next = np.max(q_next, axis=1)
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        # train eval network
        _, self.cost = self.sess.run([self.train_step, self.loss], feed_dict={
            self.s: batch_memory[:, :self.n_features], self.q_target: q_target})
        self.cost_his.append(self.cost)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

