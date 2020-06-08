import gym
import numpy as np
import cv2, random, copy

class Snake2(gym.Env):
    def __init__(self, config, **kwargs):
        self.config = config
        self.goal_position, self.grid = [], []
        self.head, self.body = [], []
        self.axis, self.direction = 0, 0
        self.grow = False
        self.episode_r, self.done = 0, False
        self.steps, self.episode = 0, 0
        self.goals_reached, self.last_goal = 0, 0

        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(4)
        if config['image_as_state']:  # Image based (CNN) or a Vector containing coordinates
            self.observation_space = gym.spaces.Box(
                shape=(config['grid_size']+2, config['grid_size']+2, 1), 
                high=1., low=-1., dtype=np.float32)
        else:                             
            self.observation_space = gym.spaces.Box(shape=(4,), high=10, low=0, dtype=np.float32)
        
    def reset(self):

        self.episode += 1
        self.steps, self.episode_r, self.goals_reached = 0, 0, 0
        self.done = False
        self.axis, self.direction = 1, 1

        self.grid = np.zeros([self.config['grid_size']+2, self.config['grid_size']+2, 1]) - 0.25
        self.grid[1:-1, 1:-1, :] = np.zeros([self.config['grid_size'], self.config['grid_size'], 1])
        self.head = [self.config['grid_size']//2, self.config['grid_size']//2]
        self.body = [[self.head[0]-x, self.head[1]] for x in reversed(range(1, self.config['initial_length']))]
        self.create_new_goal()

        return self.make_observation()

    def create_new_goal(self):
        self.goal_position = goal = list(np.random.randint(1, self.config['grid_size']+1, 2))
        if goal in self.body or goal == self.head:
            self.create_new_goal()

    def step(self, action):

        if self.grow:
            self.body.append(self.head.copy())
            self.grow = False
        else:
            self.body = self.body[1:] + [self.head.copy()]

        # Actions are defined as up, down, left, right
        action_mapping = {0: (0, -1), 1: (0, 1), 2: (1, -1), 3: (1, 1)}
        axis, direction = action_mapping[action]
        if axis != self.axis:
            self.axis, self.direction = axis, direction
        
        self.head[self.axis] += self.direction

        reward = self.collision_check()
        state = self.make_observation()
        self.episode_r += reward
        self.steps += 1

        return state, reward, self.done, {}

    def collision_check(self):
        reward = self.config['base_reward']
        if self.config['walls']:
            for axis in [0, 1]:
                if (self.head[axis]==0) or (self.head[axis]==self.config['grid_size']+1):
                    self.done = True
                    reward = self.config['collision_penalty']
        else:
            for axis in [0, 1]:
                if self.head[axis] == 0:
                    self.head[axis] = self.config['grid_size']-1
                if self.head[axis] == self.config['grid_size']-1:
                    self.head[axis] = 0
        
        if self.head in self.body:
            self.done = True
            reward = self.config['collision_penalty']
        if self.head == self.goal_position:
            reward = self.config['goal_reward']
            self.grow = True
            self.goals_reached += 1
            self.last_goal = 0
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
        self.grid[self.body[-1][0], self.body[-1][1], 0] = 0.5
        self.grid[self.body[0][0], self.body[0][1], 0] = 0.
        self.grid[self.goal_position[0], self.goal_position[1], 0] = -1.

























# if __name__ == "__main__":
    
#     from rl.baselines import get_parameters, Trainer
#     import rl.environments
#     env = TestEnv(get_parameters('TestEnv'))

#     model = Trainer('TestEnv', 'models').create_model()
#     model._tensorboard()
#     model.train()
#     print('Training done') 
#     input('Run trained model (Enter)')
#     env.create_window()
#     env.run(model)

# def run(self, model, episodes=100):
# """ Use a trained model to select actions """

# try:
#     for episode in range(episodes):
#         self.done, step = False, 0
#         state = self.reset()
#         while not self.done:
#             action = model.model.predict(state)
#             state, reward, self.done, _ = self.step(action[0])
#             print('   Episode {:2}, Step {:3}, Reward: {:.2f}, State: {}, Action: {:2}'.format(episode, step, reward, state[0], action[0]), end='\r')
#             self.render()
#             step += 1
# except KeyboardInterrupt:
#     pass

# def sample(self):
# """
# Sample random actions and run the environment
# """
# self.create_window()
# for _ in range(10):
#     self.done = False
#     state = self.reset()
#     while not self.done:
#         action = self.action_space.sample()
#         state, reward, self.done, _ = self.step(action)
#         print('Reward: {:2.3f}, state: {}, action: {}'.format(reward, state, action))
#         self.render(True)
# cv2.destroyAllWindows()