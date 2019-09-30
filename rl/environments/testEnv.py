# @author Andrius Bernatavicius, 2019
import gym
import numpy as np
import cv2, random

class TestEnv(gym.Env):

    def __init__(self, config, **kwargs):
        """
        A simple grid environment where the agent has to navigate towards a goal
        Must be a subclass of gym.Env
        """
        self.config = config['environment']

        # Grid, positions
        self.grid_size = self.config['grid_size']
        self.grid = np.zeros([self.grid_size, self.grid_size])
        self.agent_start_location = [self.grid_size//2, self.grid_size//2] # Start at the middle of the grid
        self.position = self.agent_start_location
        self.goal_position = []
        self.window_name = 'Test env'

        # Gym-related part
        self.r = 0        # Total episode reward
        self.done = False # Termination
        self.episode = 0  # Episode number
        self.steps = 0    # Current step in the episode
        self.max_steps = self.config['step_limit']
        self.goals_reached = 0

        self.create_window()

        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(4)

        if self.config['image_as_state']:  # Image based (CNN)
            self.observation_space = gym.spaces.Box(shape=(self.grid_size, self.grid_size, 1), high=1, low=0, dtype=np.uint8)
        else:                              # Vector based (MLP)
            self.observation_space = gym.spaces.Box(shape=(4,), high=10, low=0, dtype=np.uint8)
        
    def make_observation(self):
        """
        Return the environment's current state
        """
        self.position = list(np.clip(self.position, 0, self.grid_size - 1))
            # Image based (uncomment when using an image based observation space)
        if self.config['show_training']:
            self.render()
        else:
            self.render(mode='rgb_array')

        if self.config['image_as_state']:
            state = np.expand_dims(self.grid, axis=2)

        # Vector based
        else:
            state = np.array([self.position + self.goal_position], dtype=np.uint8).squeeze()
        return state

    def reset(self):
        """
        The reset method:

        Must return the current state of the environment
        """
        self.episode += 1
        print('Ep: {:5}, steps: {:3}, R: {:3.3f}'.format(self.episode, self.steps, self.r), end='\r')
        self.position = self.agent_start_location

        # Randomize goal position
        def random_location():
            self.goal_position = list(np.random.randint(0, self.grid_size - 1, 2))
            if self.goal_position == self.agent_start_location: 
                random_location() # Recursion if goal and agent overlaps at t0
        random_location()

        self.steps, self.r = 0, 0
        self.done = False

        return self.make_observation()

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        reward = -0.1  # Base reward
        self.steps += 1

        # Reward shaping
        if self.config['reward_shaping']:
            # Euclidean distance from the goal
            reward = - np.sqrt((self.position[0] - self.goal_position[0])**2 + (self.position[1] - self.goal_position[1])**2)/100

        # Perform action
        if   action==0: self.position[0] += 1
        elif action==1: self.position[0] -= 1
        elif action==2: self.position[1] += 1
        elif action==3: self.position[1] -= 1
        
        # Update the state
        self.position = list(np.clip(self.position, 0, self.grid_size - 1))
        state = self.make_observation() 

        ## Termination conditions
        if self.position == self.goal_position:
            reward += 10
            self.done = True
            self.goals_reached += 1
            print(self.goals_reached, end='\r')

        if self.steps > self.max_steps:
            self.done = True
        self.r += reward

        return state, reward, self.done, {}

    def render(self, mode='human'):
        """
        Updates the grid with current positions
        """
        self.grid = np.zeros([self.grid_size, self.grid_size])
        
        self.grid[self.position[0], self.position[1]] = 125
        self.grid[self.goal_position[0], self.goal_position[1]] = 255
        if mode=='human':
            cv2.imshow(self.window_name, self.grid.astype(np.uint8))
            cv2.waitKey(1)

        return self.grid
        
    def run(self, model, episodes=100):
        """
        Use a trained model to select actions 
        
        """
        try:
            for episode in range(episodes):
                self.done, step = False, 0
                state = self.reset()
                while not self.done:
                    action = model.model.predict(state)
                    state, reward, self.done, _ = self.step(action[0])
                    print('   Episode {:2}, Step {:3}, Reward: {:.2f}, State: {}, Action: {:2}'.format(episode, step, reward, state[0], action[0]), end='\r')
                    self.render()
                    step += 1
        except KeyboardInterrupt:
            pass
    
    def sample(self):
        """
        Sample random actions and run the environment
        """
        self.create_window()
        for _ in range(10):
            self.done = False
            state = self.reset()
            while not self.done:
                action = self.action_space.sample()
                state, reward, self.done, _ = self.step(action)
                print('Reward: {:2.3f}, state: {}, action: {}'.format(reward, state, action))
                self.render(True)
        cv2.destroyAllWindows()

    def create_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 300, 300)
                
if __name__ == "__main__":
    from rl.baselines import get_parameters, Trainer
    import rl.environments
    env = TestEnv(get_parameters('TestEnv'))

    model = Trainer('TestEnv', 'models').create_model()
    model._tensorboard()
    model.train()
    print('Training done') 
    input('Run trained model (Enter)')
    env.create_window()
    env.run(model)

   