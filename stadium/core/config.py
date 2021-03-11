import os, datetime, time, yaml
import sys, subprocess, webbrowser
from stadium.settings import GlobalConfig as settings
from stadium.environments import create_env
from stadium.core.defaults import MainConfig, EnvConfig
import stadium.core.defaults
import stable_baselines
import numpy as np
import tensorflow as tf
from tensorboard import program
tf.logging.set_verbosity(tf.logging.ERROR)
        
class InstanceConfig:
    def __init__(self, env_name=None, model=None, policy=None):
        
        self.env_name = env_name
        self.env = None
        self.model = None
        self.policy = None
        self.main = MainConfig()

        if env_name:
            self.load(env_name)

    def load_defaults(self, env):
        ## Add check for .yml files as defaults
        self.load_model(self.main.model)
        self.load_policy(self.main.policy)
        self.env = EnvConfig()

    def load_model(self, model, **kwargs):
        self.model = getattr(stadium.core.defaults, model)(**kwargs)

    def load_policy(self, policy, **kwargs):
        self.policy = getattr(stadium.core.defaults, policy)(**kwargs)

    def load_file(self, path):
        print('Loading {} config file...'.format(path))
        with open(path, 'r') as f:
            return yaml.load(f)

    def load(self, env):
        self.env_name = env
        self.env_type = self.get_env_type(env)

        for filename in [self.env_name, self.env_type, 'gym']:
            path = os.path.join(settings.CONFIG_DIR, '{}.yml'.format(filename))
            try:
                config = self.load_file(path)
                break
            except:
                print('Failed to load {} configuration file'.format(path))
            
        self.load_dict(config)
        return path

    def load_dict(self, config):
        main = config['main']
        model_type  = main['model']
        policy_type = main['policy']

        if config.get(model_type) is None:
            self.load_model(model_type)
        else:
            self.load_model(model_type, **config[model_type])
        print(config[policy_type])
        if config.get(policy_type) is None:
            self.load_policy(policy_type)
        else:
            self.load_policy(policy_type, **config[policy_type])

        self.main     = MainConfig(**config['main'])
        self.env    = config['environment']
        print(self.model)
    
    def get_env_type(self, env_name):
        for env_type, envs in settings.env_types.items():
            if env_name in envs:
                return env_type
            else:
                return 'rl'
        
    def save(self, path):
        with open(os.path.join(path, 'config.yml'), 'w') as config_file:
            yaml.dump(self.join_config(), config_file)

    def join_config(self):
        config = {
            self.main.model: self.model.__dict__,
            self.main.policy: self.policy.__dict__,
            'main': self.main.__dict__,
            'environment': self.env}

        return config

    def load_defaults(self):
        self.defaults = {}
        for defaults in ['policies', 'algorithms']:
            path = os.path.join(settings.DEFAULTS, '{}.yml'.format(defaults))
            with open(path, 'r') as f:
                config = yaml.load(f)
            self.defaults = {**self.defaults, **config}
        return config


class InstanceManager:
    """
    Creates dirs, manages envs and instances
    """
    def __init__(self, config=None, env=None, algs='baselines'):
        self.config = config
        self.env_name = ''
        self.instance_name = None

        self.env_path = None
        self.instance_path = None
        self.stop_training = False

        self.algs = algs
        self.env = None
        self.train_env = None
        self.model = None

        if env is not None:
            self.set_env(env)

    def set_env(self, env):
        self.env_name = env
        self.env_path = os.path.join(settings.TRAINED_MODELS, env)
        os.makedirs(self.env_path, exist_ok=True)
        self.config = InstanceConfig(env)
        config_path = self.config.load(env)
        self.env = self.create_env(False, n_workers=1)
        self.train_env = self.create_env(True, n_workers=1)
        return config_path

    def new_instance(self, namestamp=None):

        # Assign a unique numerical ID to an instance
        subfolders = os.listdir(self.env_path)
        numerical_ids = [0] + [int(x.split('_')[0]) for x in subfolders]
        unique_id = max(numerical_ids) + 1

        # Check if some IDs are missing (e.g. deleted)
        for num in range(max(numerical_ids)):
            if num not in numerical_ids:
                unique_id = num
                break

        if namestamp is None:
            date = datetime.datetime.now().strftime("%m-%d_%H-%M")
            namestamp = "{}_{}".format(
                self.env_name,
                date)

        if len(str(unique_id)) < 2:
            unique_id = '0' + str(unique_id)
        self.instance_name = str(unique_id) + '_' + namestamp 
        self.instance_path = os.path.join(self.env_path, self.instance_name)
        os.makedirs(self.instance_path, exist_ok=True)
        print(self.instance_path)

        self.create_model()
        self.save_instance()

    def create_env(self, multi, n_workers, **kwargs):
        return create_env(self.env_name, multi, n_workers=n_workers, **kwargs)

    def create_model(self):
        if self.algs == 'baselines':
            policy = getattr(stadium.core.baselines, self.config.main.policy)
            model_object = getattr(stable_baselines, self.config.main.model)

            model_args = dict(
                policy=policy,
                env=self.env,
                tensorboard_log=self.instance_path,
                policy_kwargs={'config': self.config.policy},
                **self.config.model.__dict__)
            
            self.model = model_object(**model_args)
        else:
            pass

    def get_models(self):
        subfolders = os.listdir(self.env_path)
        subfolders.sort()
        return subfolders

    def load_instance(self, path=None, num=None):
        """
        Loads an instance from the specified path
        """
        if num:
            subdirs = []
            for f in os.listdir(self.env_path):
                fpath = os.path.join(self.env_path, f)
                print(f)
                if os.path.isdir(fpath) and num == int(f.split('_')[0]):
                    print(fpath)
                    path = fpath
                    break

        config_path = os.path.join(path, 'config.yml')
        self.config = InstanceConfig(self.env_name)
        self.config.load_dict(self.config.load_file(config_path))
        model_file = os.path.join(path, 'model')
        model_object = getattr(stable_baselines, self.config.main.model)
        
        print(self.instance_path)
        self.model = model_object.load(model_file, env=self.env, tensorboard_log=self.instance_path)
        self.instance_path = path
        self.instance_name = os.path.split(path)[-1]
    
    def save_instance(self):
        """
        Saves the current instance (model weights and config files)
        """
        print(self.instance_path)
        self.config.save(self.instance_path)
        path = os.path.join(self.instance_path, 'model')
        self.model.save(path)

    def train(self, breakpoints=None, progress_callback=None, **kwargs):
        self.model.set_env(self.train_env)
        print('setting training environment', self.train_env)
        if breakpoints is None:
            breakpoints = settings.STEPS
        n_steps = self.config.main.steps_to_train
        n_checkpoints = n_steps//breakpoints

        train_config = dict(
            total_timesteps=breakpoints,
            tb_log_name='log_1',
            reset_num_timesteps=False)

        # Train the model and save a checkpoint every n steps
        for i in range(n_checkpoints):
            if not self.stop_training:
                self.model = self.model.learn(
                    **train_config)
                if progress_callback:
                    progress_callback.emit(breakpoints)

                self.config.main.steps_trained += breakpoints
        self.save_instance()
        self.model.set_env(self.env)


    def step(self):
        action, _ = self.model.predict(self.state)
        action_probabilities = self.model.action_probability(self.state)
        self.state, reward, done, _ = self.model.env.step(action)
        img = self.env.get_image()
            
        return img, reward, done, action_probabilities

    def callback(self, locals_, globals_):
        """
        A callback method for logging environment attributes in tensorboard

        Define them in rl/config/env_name.yml --> main: logs:
        
        Example for logging the number of steps per episode can be found in
        rl/config/TestEnv.yml
        """
        self_ = locals_['self']
        values_to_log = []
        for current_attribute_value in self.config.main.logs:
            value = np.mean(self.env.get_attr(current_attribute_value))
            values_to_log.append(tf.Summary.Value(tag=current_attribute_value, simple_value=value, ))
        summary = tf.Summary(value=values_to_log)

        return True


    def tensorboard(self, browser=True):
        # Kill current session
        self._tensorboard_kill()
        # Open the dir of the current env
        if sys.platform == 'win32':
            try:
                tb = program.TensorBoard()
                tb.configure(argv=[None, '--logdir', self.env_path])
                url = tb.launch()
                cmd = ''
            except:
                pass
        else:
            cmd = 'tensorboard --logdir {} --port 6006'.format(self.env_path) #--reload_interval 1

        try:
            DEVNULL = open(os.devnull, 'wb')
            subprocess.Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
        except:
            pass

        print('Launching tensorboard at {}'.format(self.instance_path))

    def _tensorboard_kill(self):
        """
        Destroy all running instances of tensorboard
        """
        print('Closing current session of tensorboard.')
        if sys.platform in ['win32', 'Win32']:
            try:
                os.system("taskkill /f /im tensorboard.exe")
                os.system('taskkill /IM "tensorboard.exe" /F')
            except:
                pass
        elif sys.platform in ['linux', 'linux', 'Darwin', 'darwin']:
            try:
                os.system('pkill tensorboard')
                os.system('killall tensorboard')
            except:
                pass

        else:
            print('No running instances of tensorboard.')

    def prep(self):
        self.state = self.env.reset()

        return self.env.get_image()


