import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import threading
import multiprocessing
import gym
import os
import shutil

Env_Name = 'Pendulum-v0'
Output_Graph = True
Log_Dir = './log'
N_Workers = multiprocessing.cpu_count()
Max_Ep_Steps = 400
Max_Global_Ep = 800
Global_Net_Scope = 'Global_Net'
Update_Global_Iter = 5
Gamma = 0.9
LR_A = 0.0001
LR_C = 0.001
Entropy_Beta = 0.01
Global_Runing_R = []
Global_Ep_Counter = 0

env = gym.make(Env_Name)
State_Dim = env.observation_space.shape[0]
Action_Dim = env.action_space.shape[0]
Action_Bound = [env.action_space.low, env.action_space.high]

class ACNet(object):
    def __init__(self, scope, globalAC=None):
        if scope == Global_Net_Scope:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, State_Dim], 'S')
                self._build_net()
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, State_Dim], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, Action_Dim], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'v_target')
                mu, sigma, self.v = self._build_net()
                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))
                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * Action_Bound[1], sigma + 1e-4
                normal_dist = tf.contrib.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * td
                    entropy = normal_dist.entropy()
                    self.exp_v = Entropy_Beta * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), Action_Bound[0],
                                              Action_Bound[1])
                with tf.name_scope('local_grad'):
                    self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                    self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self):
        w_init = tf.random_normal_initializer(0., 0.1)
        with tf.variable_scope('actor'):
            layer1_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='layer1_a')
            mu = tf.layers.dense(layer1_a, Action_Dim, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(layer1_a, Action_Dim, tf.nn.softplus, kernel_initializer=w_init, name='sigma')

        with tf.variable_scope('critic'):
            layer1_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='layer1_c')
            v = tf.layers.dense(layer1_c, 1, kernel_initializer=w_init, name='v')
        return mu, sigma, v

    def update_global(self, feed_dict):
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)

    def pull_global(self):
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})[0]

class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(Env_Name).unwrapped
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global Global_Ep_Counter, Global_Runing_R
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not Coord.should_stop() and Global_Ep_Counter < Max_Global_Ep:
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(Max_Ep_Steps):
                if self.name == 'W_0':
                    self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                done = True if ep_t == Max_Ep_Steps - 1 else False
                r /= 10
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % Update_Global_Iter == 0 or done:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0][0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + Gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), \
                                                          np.vstack(buffer_v_target)
                    feed_dict = {self.AC.s: buffer_s, self.AC.a_his: buffer_a, self.AC.v_target: buffer_v_target}
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()
                s = s_
                total_step += 1
                if done:
                    if len(Global_Runing_R) == 0:
                        Global_Runing_R.append(ep_r)
                    else:
                        Global_Runing_R.append(0.9 * Global_Runing_R[-1] + 0.1 * ep_r)
                    print(self.name, "Ep:", Global_Ep_Counter, "| Ep_r: %i" % Global_Runing_R[-1], )
                    Global_Ep_Counter += 1
                    break

if __name__ == "__main__":
    SESS = tf.Session()
    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        Global_AC = ACNet(Global_Net_Scope)
        workers = []
        for i in range(N_Workers):
            i_name = 'W_%i' % i
            workers.append(Worker(i_name, Global_AC))

    Coord = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())
    if Output_Graph:
        if os.path.exists(Log_Dir):
            shutil.rmtree(Log_Dir)
        tf.summary.FileWriter(Log_Dir, SESS.graph)
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    Coord.join(worker_threads)

    plt.plot(np.arange(len(Global_Runing_R)), Global_Runing_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()

