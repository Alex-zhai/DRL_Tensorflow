"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example. The RL is in RL_brain.py.

"""
import numpy as np
import tkinter as tk
import time

Grid_Num = 40
Height_Maze = 4
Width_Maze = 4
np.random.seed(1)

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_num = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(Height_Maze*Grid_Num, Width_Maze*Grid_Num))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=Height_Maze*Grid_Num, width=Width_Maze*Grid_Num)
        for c in range(0, Width_Maze*Grid_Num, Grid_Num):
            x0, y0, x1, y1 = c, 0, c, Height_Maze*Grid_Num
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, Height_Maze*Grid_Num, Grid_Num):
            x0, y0, x1, y1 = 0, r, Width_Maze*Grid_Num, r
            self.canvas.create_line(x0, y0, x1, y1)
        # create start point
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([Grid_Num * 2, Grid_Num])
        self.hell1 = self.canvas.create_rectangle(hell1_center[0] - 15, hell1_center[1] - 15,
                                                  hell1_center[0] + 15, hell1_center[1] + 15,
                                                  fill='black')
        hell2_center = origin + np.array([Grid_Num, Grid_Num * 2])
        self.hell2 = self.canvas.create_rectangle(hell2_center[0] - 15, hell2_center[1] - 15,
                                                  hell2_center[0] + 15, hell2_center[1] + 15,
                                                  fill='black')
        # create oval
        oval_center = origin + Grid_Num*2
        self.oval = self.canvas.create_oval(oval_center[0] - 15, oval_center[1] - 15,
                                            oval_center[0] + 15, oval_center[1] + 15,
                                            fill='yellow')
        # create red rect
        self.rect = self.canvas.create_rectangle(origin[0] - 15, origin[1] - 15,
                                                 origin[0] + 15, origin[1] + 15,
                                                 fill='red')
        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(origin[0] - 15, origin[1] - 15,
                                                 origin[0] + 15, origin[1] + 15,
                                                 fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:  # up action
            if s[1] > Grid_Num:
                base_action[1] -= Grid_Num
        elif action == 1:  # down
            if s[1] < (Height_Maze - 1) * Grid_Num:
                base_action[1] += Grid_Num
        elif action == 2:   # right
            if s[0] < (Width_Maze - 1) * Grid_Num:
                base_action[0] += Grid_Num
        elif action == 3:   # left
            if s[0] > Grid_Num:
                base_action[0] -= Grid_Num
        self.canvas.move(self.rect, base_action[0], base_action[1])
        s_next = self.canvas.coords(self.rect)
        if s_next == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif s_next in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        return s_next, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()