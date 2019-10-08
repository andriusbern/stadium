# @author: Andrius Bernatavicius, 2019

from stable_baselines.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv, VecNormalize
from stable_baselines.common.schedules import LinearSchedule, linear_interpolation
from collections import deque
import numpy as np
import os, yaml, sys, subprocess, webbrowser, time, datetime, random, copy
import cv2
import stable_baselines, gym, rl
import rl.settings as settings
import tensorflow as tf
import glob

# Error suppression (numpy 1.16.2, tensorflow 1.13.1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.info('TensorFlow')
tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.info('TensorFlow')


def get_env_type(env_name):
    """
    Get the type of environment from the env_name string
    """
    try:
        env = gym.make(env_name)
        del env
        return 'gym'
    except:
        return 'rl'

def create_env(env_name, config=None, n_workers=1, image_based=True, **kwargs):
    """
    Parses the environment to correctly return the attributes based on the spec and type
    Creates a corresponding vectorized environment
    """

    def make_rl(**kwargs):
        """
        Decorator for custom RL environments
        """
        def _init():
            env_obj = getattr(rl.environments, env_name)
            env = env_obj(config)
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
    
    if config is not None:
        n_workers = config['main']['n_workers']
    mapping = {'gym': make_gym, 'rl': make_rl}
    env_type = get_env_type(env_name)
    env_decorator = mapping[env_type]
    vectorized_decorator = [env_decorator(rank=x) for x in range(n_workers)]

    # Parallelize
    if n_workers > 1:
        method = 'spawn' if sys.platform == 'win32' else 'forkserver'
        vectorized = SubprocVecEnv(vectorized_decorator, start_method=method)
    else: # Non multi-processing env
        vectorized = DummyVecEnv(vectorized_decorator)

    # Frame-stacking for CNN based environments
    if 'frame_stack' in config['main'].keys():
        if config['main']['frame_stack'] != 0:
            vectorized = VecFrameStack(vectorized, n_stack=config['main']['frame_stack'])
    if 'normalize' in config['main'].keys():
            vectorized = VecNormalize(vectorized, clip_obs=1, clip_reward=1)

    return vectorized

def get_parameters(env_name, model_path=None, config_name=None, config_location=None):
    """
    Method for getting the YAML config file of the RL model, policy and environment
    Get config by prioritizing:
        1. Specific config file: /config/[name].yml
        2. From model's directory (in case of loading) /trained_models/_/_/_/parameters.yml
        3. /config/[env_name].yml
        4. /config/[env_type].yml
        5. /config/defaults.yml
    """
    env_type = get_env_type(env_name)
    env_params = os.path.join(settings.CONFIG, env_name+'.yml')
    if config_location is not None:
        path = config_location
    else:
        if config_name is not None:
            path = os.path.join(settings.CONFIG, config_name + '.yml')
        elif model_path is not None:
            path = os.path.join(model_path, 'config.yml')
        elif os.path.isfile(env_params):
            path = env_params
        else:
            path = os.path.join(settings.CONFIG, env_type + '.yml')

    with open(path, 'r') as f:
        config = yaml.load(f)
    print('\nLoaded config file from: {}\n'.format(path))

    return config

class Trainer(object):
    """
    Wrapper for stable_baselines library
    """

    def __init__(self, env, subdir='', model_from_file=None):

        self.config = None
        self.env = None
        self.model = None
        self.name = None

        self.env_name = env
        self._env_type = get_env_type(self.env_name)
        self.date = datetime.datetime.now().strftime("%m-%d_%H-%M")
        self._env_path = os.path.join(settings.TRAINED_MODELS, env, subdir)
        self._model_path = None
        self.reloaded = False
        self.done = True
        self.test_state = None
         
    def load_model(self, num=None, config_file=None, latest=False, path=None):
        """
        Load a saved model either from the 
        """
        print('Loading path {}'.format(self._env_path))
        assert os.path.isdir(self._env_path), 'Path {} does not exist.'.format(self._env_path)

        folder_list = glob.glob(self._env_path + '/*') 
        if latest:
            model_path = max(folder_list, key=os.path.getctime)
        else:
            for folder in folder_list:
                print(folder)
                if int(folder.split('\\')[-1].split('_')[0]) == num:
                    model_path = folder
                    if not os.path.isfile(os.path.join(model_path, 'model.pkl')):
                        model_path = model_path[:-1] + '1'
                    print('Model path:', model_path)
                    break  

        self._model_path = model_path
        self.config = get_parameters(self.env_name, self._model_path, config_name=config_file)
        self.n_steps = self.config['main']['n_steps']
        model_file = os.path.join(model_path, 'model.pkl')
        model_object = getattr(stable_baselines, self.config['main']['model'])
        self._unique_model_identifier = model_path.split('\\')[-1]
        print('Unique path: {}'.format(self._unique_model_identifier))
        
        self.create_env()
        self.model = model_object.load(model_file[:-4], env=self.env, tensorboard_log=self._env_path)
        self.reloaded = True
        print('Loading model file {}'.format(model_file))

        return self

    def create_model(self, config_file=None, dataset=None, config_location=None, name=None):
        """
        Creates a new RL Model
        """
        
        self.name = name
        if config_file is None:
            args = dict(env_name=self.env_name)
            args['config_location'] = config_location
            c = self.config = get_parameters(**args)
        else:
            c = self.config = config_file

        self.n_steps = self.config['main']['n_steps']
        self.create_env()

        model_name    = c['main']['model']
        model_params  = c['models'][model_name]
        policy_name   = c['main']['policy']
        policy_params = c['policies'][policy_name]
        print('\nCreating {} model...'.format(model_name))

        self.policy = self._get_policy(policy_name)
        model_object = getattr(stable_baselines, model_name)

        model_args = dict(
            policy=self.policy, 
            env=self.env,
            tensorboard_log=self._env_path,
            **model_params)
        
        # DDPG Model creation
        if 'DDPG' in model_name:
            from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec, NormalActionNoise
            n_actions = self.env.action_space.shape[0]
            model_args['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
        
        if 'Custom' in policy_name:
            if 'DQN' in model_name:
                self.policy = model_args['policy'] = self._get_policy('CustomDQNPolicy')
                model_args['policy_kwargs'] = {**c['policies']['CustomDQNPolicy']}
            else:
                model_args['policy_kwargs'] = {'params': policy_params}
                
        self.model = model_object(**model_args)

        return self

    def _get_policy(self, policy_name):
        """
        Returns a corresponding policy object either from stable_baselines
        """
        if hasattr(stable_baselines.common.policies, policy_name):
            return getattr(stable_baselines.common.policies, policy_name)
        else:
            return getattr(rl.baselines, policy_name)

    def create_env(self):
        """
        Parses the environment to correctly return the attributes based on the spec and type
        Creates a corresponding vectorized environment
        """
        print('Creating {} Environment...\n'.format(self.env_name))
        self.env = create_env(self.env_name, self.config)

    # Directory management
    def _create_model_dir(self):
        """
        Creates a unique subfolder in the environment directory for the current trained model
        """
        # Create the environment specific directory if it does not exist
        if not os.path.isdir(self._env_path):
            os.makedirs(self._env_path)

        # Get the unique id [N] of the directory ../trained_models/env_type/env/[N]_MODEL/...
        try:
            num = max([int(x.split('_')[0]) for x in os.listdir(self._env_path)]) + 1 # Find the highest id number of current trained models
        except:
            num = 0

        c = self.config['main']
        if self.name is not None:
            dir_name = self.name
        else:
            # Modify this based on what's relevant for identifying the trained models
            dir_name = "{}_{}_{}_{}_{}".format(c['model'], c['policy'], c['n_steps'], c['n_workers'], self.date) # Unique stamp

        self._unique_model_identifier = str(num) + '_' + dir_name #+ '_1' # Unique identifier of this model
        self._model_path = os.path.join(self._env_path, self._unique_model_identifier) # trained_models/env_type/env/trainID_uniquestamp
        os.makedirs(self._model_path, exist_ok=True)

    def _delete_incomplete_models(self):
        """
        Deletes directories that do not have the model file saved in them
        """
        import shutil
        count = 0
        for model_folder in os.listdir(self._env_path):
            path = os.path.join(self._env_path, model_folder)
            files = os.listdir(path)
            if 'model.pkl' not in files:
                shutil.rmtree(path)
                count += 1
        print('Cleaned directory {} and removed {} folders.'.format(self._env_path, count))

    def _save(self):
        self.model.save(os.path.join(self._model_path+'_1', 'model'))
        # Save config
        with open(os.path.join(self._model_path+'_1', 'config.yml'), 'w') as f:
            yaml.dump(self.config, f, indent=4, sort_keys=False, line_break=' ')

    def _tensorboard(self, env_name=None):
        # Kill current session
        self._tensorboard_kill()

        # Open the dir of the current env
        cmd = 'tensorboard.exe --logdir ' + self._env_path
        print('Launching tensorboard at {}'.format(self._env_path))
        DEVNULL = open(os.devnull, 'wb')
        subprocess.Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
        time.sleep(2)
        webbrowser.open_new_tab(url='http://localhost:6006/#scalars&_smoothingWeight=0.995')
    
    def _tensorboard_kill(self):
        """
        Destroy all running instances of tensorboard
        """
        print('Closing current session of tensorboard.')
        if sys.platform == 'win32':
            os.system("taskkill /f /im  tensorboard.exe")
        elif sys.platform == 'linux':
            os.system('pkill tensorboard')
        else:
            print('No running instances of tensorboard.')

    def _check_env_status(self):
        """
        In case one of the vectorized environments breaks - recreate it.
        """
        try:
            self.env.reset()
        except BrokenPipeError as e:
            self.create_env()
            self.model.set_env(self.env)
            print(e, '\n BPE: Recreating environment...')
        except EOFError as e:
            self.create_env()
            self.model.set_env(self.env)
            print(e, '\n EOF: Recreating environment...')

    def callback(self, locals_, globals_):
        """
        A callback method for logging environment attributes in tensorboard

        Define them in rl/config/env_name.yml --> main: logs:
        
        Example for logging the number of steps per episode can be found in
        rl/config/TestEnv.yml
        """
        self_ = locals_['self']
        # Log additional tensor
        if not self_.is_tb_set:
            with self_.graph.as_default():
                tf.summary.scalar('value_target', tf.reduce_mean(self_.value_target))
                self_.summary = tf.summary.merge_all()
            self_.is_tb_set = True
        # Log scalar threshold (here a random variable)
        values_to_log = []
        for current_attribute_value in self.config['main']['logs']:
            value = np.mean(self.env.get_attr(current_attribute_value))
            values_to_log.append(tf.Summary.Value(tag=current_attribute_value, simple_value=value, ))
        summary = tf.Summary(value=values_to_log)
        locals_['writer'].add_summary(summary, self_.num_timesteps)

        return True

    def _save_env_attribute(self, attribute):
        """
        Obtains and saves anvironment specific atributes in a text file
        (Only one of the environments in case they're running in parallel)
        """
        try:
            data = self.env.get_attr(attribute)
            with open(os.path.join(self._model_path, attribute + '.log'), 'a') as f:            
                for item in data:
                    f.write('%f\n' % item[0])
        except:
            print('Attribute does not exist.')

    #########################################
    # Methods for training and testing
    def train(self, steps=None):
        """
        Train method
        """
        if not self.reloaded:
            self._create_model_dir()
        self._check_env_status()
        try:
            save_every = self.config['main']['save_every']
            n_steps = steps if steps is not None else self.n_steps
            n_checkpoints = n_steps//save_every
            self.model.is_tb_set = True
            config = dict(
                total_timesteps=save_every,
                tb_log_name=self._unique_model_identifier,
                reset_num_timesteps=True,
                seed=None)

            # Train the model and save a checkpoint every n steps
            print('CTRL + C to stop the training and save.\n')
            for i in range(n_checkpoints):
                self.reloaded = True
                self.model = self.model.learn(callback=self.callback, **config)
                self._save()
        except KeyboardInterrupt:
            self._save()
            print('Done.')

    def run(self, episodes=20, delay=1, render=True, image_based=False):
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
                        self.model.env.render()
                    step += 1
        except KeyboardInterrupt:
            pass

        
if __name__ == "__main__":
    env = 'MountainCarContinuous-v0'
    b = Trainer(env)
    b.create_model()
    b.run()
