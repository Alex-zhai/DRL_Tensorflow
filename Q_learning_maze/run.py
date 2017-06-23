from Q_learning_maze.maze_env import Maze
from Q_learning_maze.agent import QLearningTable

def update():
    for episode in range(100):
        ob = env.reset()
        while True:
            env.render()
            action = RL.choose_action(str(ob))
            next_ob, reward, done = env.step(action)
            RL.learn(str(ob), action, reward, str(next_ob))
            ob = next_ob
            if done:
                break
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.action_num)))
    env.after(100, update)
    env.mainloop()

