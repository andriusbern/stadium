import gym
import numpy as np

class TestEnv(gym.Env):
    def __init__(self, config, **kwargs):
        self.config = config
        self.position, self.goal_position = [], []
        self.grid = []
        self.episode_r, self.done = 0, False
        self.steps, self.episode = 0, 0

        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(4)
        if config['image_as_state']:  # Image based (CNN) or a Vector containing coordinates
            self.observation_space = gym.spaces.Box(shape=(config['grid_size'], config['grid_size'], 1), high=1, low=0, dtype=np.uint8)
        else:                             
            self.observation_space = gym.spaces.Box(shape=(4,), high=10, low=0, dtype=np.uint8)
        self.create_window() if config['show_training'] else None
        
    def reset(self):

        self.episode += 1
        self.steps, self.episode_r = 0, 0
        self.done = False
        print('Ep: {:5}, steps: {:3}, R: {:3.3f}'.format(self.episode, self.steps, self.episode_r), end='\r')

        self.grid = np.ones([self.config['grid_size']+2, self.config['grid_size']+2])
        self.grid[1:-1, 1:-1] = np.zeros([self.config['grid_size'], self.config['grid_size']])

        def randomize_locations():
            self.goal_position = list(np.random.randint(1, self.config['grid_size'], 2))
            self.position      = list(np.random.randint(1, self.config['grid_size'], 2))
            if self.goal_position == self.position: 
                randomize_locations()
        randomize_locations()

        return self.make_observation()

    def step(self, action):

        self.steps += 1
        self.grid[self.position[0], self.position[1]] = .25

        # Perform action, update agent's position
        action_mapping = {0: (0, 1), 1: (0, -1), 2: (1, 1), 3: (1, -1)}
        axis, increment = action_mapping[action]
        self.position[axis] += increment
        state = self.make_observation() 

        # Rewards, shaping
        reward = -0.1 
        if self.config['reward_shaping']:
            reward -= np.sqrt((self.position[0] - self.goal_position[0])**2 + 
                              (self.position[1] - self.goal_position[1])**2)/100
        
        ## Termination conditions
        if self.steps > self.config['step_limit']:
            self.done = True
        if self.position == self.goal_position:
            reward += self.config['goal_reward']
            self.done = True
        self.episode_r += reward

        return state, reward, self.done, {}

    def render(self, mode='human'):

        self.grid[self.position[0], self.position[1]] = .5
        self.grid[self.goal_position[0], self.goal_position[1]] = 1.
        if mode=='human' and self.config['show_training']:
            # cv2.imshow(self.window_name, self.grid.astype(np.uint8))
            # cv2.waitKey(1)
            pass
        else:
            return self.grid
            
    def make_observation(self):
        """Return the environment's current state"""

        self.position = list(np.clip(self.position, 1, self.config['grid_size']))
        mode = 'human' if self.config['show_training'] else 'rgb_array'
        self.render(mode=mode)

        if self.config['image_as_state']:
            state = np.expand_dims(self.grid[1:-1, 1:-1], axis=2)
        else:
            state = np.array([self.position + self.goal_position], dtype=np.uint8).squeeze() / self.config['grid_size']

        return state
        
    # def create_window(self):

    #     self.window_name = 'Test env{}'.format(random.randint(1,10000))
    #     cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow(self.window_name, 300, 300)
























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