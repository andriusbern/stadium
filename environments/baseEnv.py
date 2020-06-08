import gym

class BaseEnv(gym.Env):
    def __init__(self, config, **kwargs):
        self.config = config

    def config_check(self, config):
        pass

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
        pass
    
    def reset(self):
        """
        The reset method:

        Must return the current state of the environment
        """
        return 0

    def render(self):
        pass
