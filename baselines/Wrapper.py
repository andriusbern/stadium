# @author: Andrius Bernatavicius, 2019

from stable_baselines.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv, VecNormalize
from stable_baselines.common.schedules import LinearSchedule, linear_interpolation
import tensorflow as tf
import numpy as np
import os, yaml, sys, subprocess, webbrowser, time, datetime, random, copy
import stable_baselines, gym, rl
import glob
from stable_baselines import PPO2
from rl.settings import GlobalConfig as settings
import rl.environments
import rl.baselines
try:
    import vizdoomgym
    import rlif
    import nao_rl
    import rusher.environments
    import rlif.environments
    import nao_rl.environments
except:
    pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.info('TensorFlow')
tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.info('TensorFlow')


def create_env(env_name, config_dict=None, n_workers=1, **kwargs):
    """
    Parses the environment to correctly return the attributes based on the spec and type
    Creates a corresponding vectorized environment
    """

    def make_custom(config_dict, **kwargs):
        """
        Decorator for custom RL environments
        """
        def _init():
            try:
                env_obj = getattr(rl.environments, env_name)
            except:
                env_obj = getattr(rlif.environments, env_name)
                # env_obj = getattr(rusher.environments, env_name)
            env = env_obj(config_dict)
            return env
        return _init

    def make_vrep(env_name, **kwargs):
        """
        Decorator for custom RL environments
        """
        def _init():
            try:
                env = nao_rl.make(env_name)
            except:
                pass
            return env
        return _init

    def make_gym(rank, seed=0, **kwargs):
        """
        Decorator for gym environments
        """
        def _init():
            env = gym.make(env_name)
            env.seed(seed + rank)
            return env
        return _init
    
    # Check whether it's custom environment
    try:
        env = gym.make(env_name)
        del env
        vectorized_decorator = [make_gym(rank=x) for x in range(n_workers)]
        is_custom = False
    except:
        if env_name in settings.envs['vrep']:
            vectorized_decorator = [make_vrep(env_name) for x in range(n_workers)]
        else:
            vectorized_decorator = [make_custom(config_dict) for x in range(n_workers)]
        is_custom = True

    # Parallelize
    if n_workers > 1:
        vectorized = SubprocVecEnv(vectorized_decorator)
    else: # Non multi-processing env
        vectorized = DummyVecEnv(vectorized_decorator)

    return vectorized, is_custom


class Trainer(object):
    """
    Wrapper for stable_baselines library
    """

    def __init__(self):
        self.env = None
        self.model = None
        self.name = None
        self.reloaded = False
        self.is_custom = False
        self.state = None
        self.stop = False
        self.path = None
        self.config = None
        self.steps_trained = 0
        self.current_checkpoint = 0
    
    def create_model(self, config, path):
        """
        Creates a new RL Model
        """
        self.config = config
        self.env, self.is_custom = create_env(
            env_name=config.env_name,
            config_dict=config.env,
            n_workers=config.trainer['n_workers'])

        policy = self._get_policy_class(config.policy_type)
        model_object = getattr(stable_baselines, config.model_type)
        model_args = dict(
            policy=policy,
            env=self.env,
            tensorboard_log=path,
            **config.model)

        if 'Custom' in config.policy_type:
            model_args['policy_kwargs'] = {'params': config.policy}

        self.model = model_object(**model_args)
        self.path = path
        self.save_model()
        self.steps_trained = 0

    def load_model(self, path, config):
        """
        Load a saved model either from the 
        """
        self.config = config
        model_file = os.path.join(path, 'model.pkl')
        model_object = getattr(stable_baselines, config.model_type)
        self.env, self.is_custom = create_env(
            env_name=config.env_name, 
            config_dict=config.env,
            n_workers=config.trainer['n_workers'])
        self.model = model_object.load(model_file[:-4], env=self.env, tensorboard_log=settings.ENV_DIR)
        self.reloaded = True
        self.path = path
        self.steps_trained = 0

    def save_model(self):
        path = os.path.join(self.path, 'model')
        self.model.save(path)
        self.model.save(path+str(self.current_checkpoint))
        self.current_checkpoint += 1

    def create_env(self, n_workers=1):
        self.env, self.is_custom = create_env(
            env_name=self.config.env_name, 
            config_dict=self.config.env,
            n_workers=n_workers)
        return self.env


    #########################################
    # Methods for training and testing

    def train(self, breakpoints=None):
        """
        Train method
        """
        # self._check_env_status()
        if breakpoints is None:
            breakpoints = settings.STEPS
        try:
            n_steps = self.config.trainer['n_steps']
            n_checkpoints = n_steps//breakpoints
            self.model.is_tb_set = True

            train_config = dict(
                total_timesteps=breakpoints,
                tb_log_name='log_1',
                reset_num_timesteps=False,
                seed=None)
            if self.config.trainer.get('steps_trained') is None:
                self.config.trainer['steps_trained'] = 0

            

            # Train the model and save a checkpoint every n steps
            print('CTRL + C to stop the training and save.\n')
            for i in range(n_checkpoints):
                if not self.stop:
                    self.reloaded = True
                    self.model = self.model.learn(callback=self.callback, **train_config)
                    self.save_model()
        except KeyboardInterrupt:
            self.save_model()
            print('Done.')
        self._check_env_status()

    def prep(self):
        # self._check_env_status()
        self.state = self.model.env.reset()

        return self.get_image()

    def step(self):
        action, _ = self.model.predict(self.state)
        prob = self.model.action_probability(self.state)
        self.state, reward, self.done, _ = self.model.env.step(action)
        img = self.get_image()
        
        return img, prob, reward, self.done

    def get_image(self):
        env = self.config.env_name
        if env in settings.envs['vizdoom']:
            img = self.state.squeeze()[:, :, 0:3]
        elif env in settings.envs['grid']:
            img = self.model.env.get_attr('grid')
            img = np.stack(img)
        elif env in settings.envs['rlif']:
            img = self.state.squeeze()
        return img
        

    def run(self, episodes=20, delay=1, render=True):
        """
        Run a small test loop
        """
        self._check_env_status()

        try:
            for episode in range(episodes):
                self.done, step = [False], 0
                self.test_state = self.model.env.reset()
                while not any(self.done):
                    action, _ = self.model.predict(self.test_state)
                    self.test_state, reward, self.done, _ = self.model.env.step(action)
                    print('   Episode {:2}, Step {:3}, Reward: {:.2f}'.format(episode, step, reward[0]), end='\r')
                    if render:
                        self.model.env.render(close=True)
                    step += 1
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass

    def _get_policy_class(self, policy_name):
        """
        Returns a corresponding policy object either from stable_baselines
        """
        if hasattr(stable_baselines.common.policies, policy_name):
            return getattr(stable_baselines.common.policies, policy_name)
        else:
            return getattr(rl.baselines, policy_name)

    def callback(self, locals_, globals_):
        """
        A callback method for logging environment attributes in tensorboard

        Define them in rl/config/env_name.yml --> main: logs:
        
        Example for logging the number of steps per episode can be found in
        rl/config/TestEnv.yml
        """
        self_ = locals_['self']
        if not self_.is_tb_set:
            with self_.graph.as_default():
                tf.summary.scalar('value_target', tf.reduce_mean(self_.value_target))
                self_.summary = tf.summary.merge_all()
            self_.is_tb_set = True
        # Log scalar threshold (here a random variable)
        values_to_log = []
        for current_attribute_value in self.config.trainer['logs']:
            value = np.mean(self.env.get_attr(current_attribute_value))
            values_to_log.append(tf.Summary.Value(tag=current_attribute_value, simple_value=value, ))
        summary = tf.Summary(value=values_to_log)
        locals_['writer'].add_summary(summary, self_.num_timesteps)

        return True

    def _check_env_status(self):
        """
        In case one of the vectorized environments breaks - recreate it.
        """
        self.env, self.is_custom = create_env(self.config.env_name, self.config.env, n_workers=self.config.trainer['n_workers'])
        self.model.set_env(self.env)

