import gym
from Double_DQN.agent import DoubleDQN
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 11

sess = tf.Session()
with tf.variable_scope('DQN'):
    DQN = DoubleDQN(n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE, e_greedy_increment=0.001,
                    double_q=False, sess=sess)
with tf.variable_scope('DDQN'):
    DDQN = DoubleDQN(n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE, e_greedy_increment=0.001,
                     double_q=True, sess=sess, out_graph=True)

sess.run(tf.global_variables_initializer())

def train(RL):
    total_step = 0
    observation = env.reset()
    while True:
        # if total_steps - MEMORY_SIZE > 8000: env.render()
        action = RL.choose_action(observation)
        f_action = (action - (ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)
        observation_, reward, done, info = env.step([f_action])
        reward /= 10
        RL.store_transition(observation, action, reward, observation_)
        if total_step > MEMORY_SIZE:
            RL.learn()
        if total_step - MEMORY_SIZE > 20000:
            break
        observation = observation_
        total_step += 1
    return RL.q

q_DQN = train(DQN)
q_DDQN = train(DoubleDQN)

plt.plot(np.array(q_DQN), c='r', label='natural')
plt.plot(np.array(q_DDQN), c='b', label='double')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('training steps')
plt.grid()
plt.show()