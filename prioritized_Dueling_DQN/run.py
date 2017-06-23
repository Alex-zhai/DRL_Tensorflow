import gym
import numpy as np
import matplotlib.pyplot as plt
from prioritized_Dueling_DQN.PrioritzedDuelingDQN import DuelingDQNPrioritizedReplay

# 设置运行环境
env = gym.make('MountainCar-v0')
env = env.unwrapped
env.seed(21)

# 设置参数
N_A = env.action_space.n
N_S = env.observation_space.shape[0]
MEMORY_CAPACITY = 50000
MEMORY_SIZE = 5000
TARGET_REP_ITER = 2000
MAX_EPISODES = 900
E_GREEDY = 0.95
E_INCREMENT = 0.00001
GAMMA = 0.99
LR = 0.0001
BATCH_SIZE = 32
HIDDEN = [100, 50]
RENDER = True


def train(RL):
    total_steps = 0
    steps = []
    episodes = []
    for i_episode in range(20):
        observation = env.reset()
        while True:
            # env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            if done:
                reward = 10

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()

            if done:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            observation = observation
            total_steps += 1
    return np.vstack((episodes, steps))

if __name__ == "__main__":

    RL = DuelingDQNPrioritizedReplay(
        n_actions=N_A, n_features=N_S, learning_rate=LR, e_greedy=E_GREEDY, reward_decay=GAMMA,
        hidden=HIDDEN, batch_size=BATCH_SIZE, replace_target_iter=TARGET_REP_ITER,
        memory_size=MEMORY_CAPACITY, e_greedy_increment=E_INCREMENT, )

    his_agent = train(RL)
    # compare based on first success
    plt.plot(his_agent[0, :], his_agent[1, :] - his_agent[1, 0], c='b', label='Prioritized_dueling_DQN')
    plt.legend(loc='best')
    plt.ylabel('total training time')
    plt.xlabel('episode')
    plt.grid()
    plt.show()

