import gym
import numpy as np

class Snake:
    def __init__(self, grid_size=10, initial_length=5, goal_reward=1., base_reward=-0.02, collision_penalty=-2., walls=True, **kwargs):
        self.metadata = None
        self.size = grid_size
        self.initial_length = initial_length
        self.goal_reward = goal_reward
        self.base_reward = base_reward
        self.collision_penalty = collision_penalty
        self.walls = walls
        self.grid = None
        self.reset()
        
        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            shape=(self.size + 2, self.size + 2, 1), 
            high=1., low=-1., dtype=np.float32)

    def reset(self):
        self.steps, self.goals_reached = 0, 0
        self.done, self.grow = False, False
        self.axis, self.direction = 0, 1
        size = self.size

        self.grid = np.zeros([size + 2, size + 2, 1]) - 0.25
        self.grid[1:-1, 1:-1, :] = np.zeros([size, size, 1])
        self.head = [size // 2, size // 2]
        self.body = [[self.head[0]-x, self.head[1]] for x in reversed(range(1, self.initial_length))]
        self.create_new_goal()

        return self.render()

    def create_new_goal(self):
        self.goal_position = goal = list(np.random.randint(1, self.size + 1, 2))
        if goal == self.head or goal in self.body:
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
        state = self.render()
        self.steps += 1

        return state, reward, self.done, {}

    def collision_check(self):
        reward = self.base_reward
        if self.walls:
            for axis in [0, 1]:
                if (self.head[axis]==0) or (self.head[axis]==self.size + 1):
                    self.done = True
                    reward = self.collision_penalty
        else:
            for axis in [0, 1]:
                right_bound = self.size - 1
                self.head[axis] = (self.head[axis]+right_bound)
                if self.head[axis] == 0:
                    self.head[axis] = right_bound
                if self.head[axis] == right_bound:
                    self.head[axis] = 0
        
        if self.head in self.body:
            self.done = True
            reward = self.collision_penalty
        if self.head == self.goal_position:
            reward = self.goal_reward
            self.grow = True
            self.goals_reached += 1
            self.create_new_goal()

        return reward

    def render(self, mode='human', **kwargs):

        self.grid[self.goal_position[0], self.goal_position[1], 0] = -1.
        self.grid[self.head[0], self.head[1], 0] = 1. # Draw head
        self.grid[self.body[-1][0], self.body[-1][1], 0] = 0.5 # Highlight last segment
        self.grid[self.body[0][0], self.body[0][1], 0] = 0. # Mask last segment in prev frame
        return self.grid


if __name__ == "__main__":
    e = Snake2()




















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