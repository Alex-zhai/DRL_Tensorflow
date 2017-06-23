"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

"""
import numpy as np
import pandas as pd
import time

np.random.seed(2)

State_Nums = 6
Actions = ['left', 'right']
Epsilon = 0.9
Gamma = 0.9
Alpha = 0.1
Max_Episode = 15
Fresh_time = 0.3

def create_q_table(states_nums, actions):
    q_table = pd.DataFrame(np.zeros((states_nums, len(actions))), columns=actions)
    return q_table

def choose_action(state, q_table):
    state_action = q_table.iloc[state, :]
    if (np.random.uniform() > Epsilon) or (state_action.all() == 0):
        return np.random.choice(Actions)
    else:
        return state_action.argmax()

def step_in_env(state, action):
    if action == 'right':
        if state == State_Nums - 2:
            r = 1
            s_next = 'terminal'
        else:
            s_next = state + 1
            r = 0
    else:
        r = 0
        if state == 0:
            s_next = 0
        else:
            s_next = state - 1
    return s_next, r

def update_env(state, episode, step_counter):
    env_list = ['-']*(State_Nums - 1) + ['T']
    if state == 'terminal':
        interaction = 'Episode %s: total_step = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(Fresh_time)

def q_update():
    q_table = create_q_table(State_Nums, Actions)
    for episode in range(Max_Episode):
        step_counter = 0
        state = 0
        is_terminal = False
        update_env(state, episode, step_counter)
        while not is_terminal:
            action = choose_action(state, q_table)
            s_next, r = step_in_env(state, action)
            q_pred = q_table.ix[state, action]
            if s_next == 'terminal':
                q_target = r
                is_terminal = True
            else:
                q_target = r + Gamma * q_table.iloc[s_next, :].max()
            q_table.ix[state, action] += Alpha * (q_target - q_pred)
            state = s_next
            update_env(state, episode, step_counter+1)
            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = q_update()
    print('\r\nQ-table:\n')
    print(q_table)



