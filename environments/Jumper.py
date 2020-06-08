import gym
import numpy as np
import cv2, random, copy

class Jumper(gym.Env):
    def __init__(self, config, **kwargs):
        self.config = config
        self.goal_position, self.grid = [], []
        self.head, self.body = [], []
        self.axis, self.direction = 0, 0
        self.grow = False
        self.episode_r, self.done = 0, False
        self.steps, self.episode = 0, 0
        self.goals_reached = 0

        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(3)
        if config['image_as_state']:  # Image based (CNN) or a Vector containing coordinates
            self.observation_space = gym.spaces.Box(
                shape=(config['grid_size']+2, config['grid_size']+2, 1), 
                high=1., low=-1., dtype=np.float32)
        else:                             
            self.observation_space = gym.spaces.Box(shape=(4,), high=10, low=0, dtype=np.uint8)
        # self.create_window() if config['show_training'] else None
        
    def reset(self):

        self.episode += 1
        # print('Ep: {:5}, steps: {:3}, R: {:3.3f}'.format(self.episode, self.steps, self.episode_r), end='\r')
        self.steps, self.episode_r, self.goals_reached = 0, 0, 0
        self.done = False
        self.axis, self.direction = 1, 0

        self.grid = np.zeros([self.config['grid_size'], self.config['grid_size'], 1])
        self.grid[1:-1, 1:-1, :] = np.zeros([self.config['grid_size'], self.config['grid_size'], 1])
        self.head = [self.config['grid_size']//2, self.config['grid_size']//2]
        self.body = [[self.head[0]-x, self.head[1]] for x in reversed(range(1, self.config['initial_length']))]
        self.create_new_goal()

        return self.make_observation()

    def create_new_obstacle(self):
        height = np.random.randint(1, 3)
        self.grid[-1, self.config['grid_size']-height:self.config.grid_size , 0] = -1.

        # self.obstacles.append(obstacle)
        



    def step(self, action):

        if self.grow:
            self.body.append(self.head.copy())
            self.grow = False
        else:
            self.body = self.body[1:] + [self.head.copy()]

        # Actions are defined as [0: do nothing; 1: go right; 2: go left]
        action_mapping = {0: (0, 0), 1: (1, 0), 2: (1, 1)}
        swap_axis, swap_direction = action_mapping[action]
        self.axis = (self.axis + swap_axis)%2
        self.direction = (self.direction + swap_direction)%2
        self.head[self.axis] += self.direction * 2 - 1

        state = self.make_observation()
        reward = self.collision_check()
        self.episode_r += reward
        self.steps += 1

        return state, reward, self.done, {}

    def collision_check(self):
        reward = self.config['base_reward']
        for axis in [0, 1]:
            if (self.head[axis]==0) or (self.head[axis]==self.config['grid_size']) or (self.head in self.body):
                self.done = True
                reward = self.config['collision_penalty']
        if self.head == self.goal_position:
            reward = self.config['goal_reward']
            self.grow = True
            self.goals_reached += 1
            self.create_new_goal()

        return reward

    def make_observation(self):

        mode = 'human' if self.config['show_training'] else 'rgb_array'
        self.render(mode=mode)
        if self.config['image_as_state']:
            state = self.grid
        else:
            state = np.array([self.head + self.goal_position], dtype=np.uint8).squeeze() / self.config['grid_size']

        return state

    def render(self, mode='human', state=None):

        self.grid[self.head[0], self.head[1], 0] = 1.
        self.grid[self.body[-1][0], self.body[-1][1], 0] = 0.75
        self.grid[self.body[0][0], self.body[0][1], 0] = 0.
        self.grid[self.goal_position[0], self.goal_position[1], 0] = -1.

        if mode=='human' and self.config['show_training']:
            cv2.imshow(self.window_name, self.grid[1:-1, 1:-1, 0])
            cv2.waitKey(1)
        else:
            pass
            # return np.dstack([self.grid, self.grid, self.grid])
        
    def create_window(self):

        self.window_name = 'Test env{}'.format(random.randint(1,10000))
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 300, 300)






















