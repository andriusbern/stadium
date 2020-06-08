import os, datetime, time, yaml
import sys, subprocess, webbrowser
from rl.settings import GlobalConfig as settings
from rl.baselines import Trainer

class ConfigManager:
    def __init__(self, env_name=None):

        self.trainer = {}
        self.defaults = {}
        ## Steps, logs, save every, etc, trained for #, n_retrained

        self.model_type = None
        self.model = {}

        self.env_name = None
        self.env_type = ''
        self.env = {}

        self.policy_type = None
        self.policy = {}

        self.load_defaults()
        if env_name is not None:
            self.load(env_name)

    def load(self, env):
        self.env_name = env
        self.env_type = self.get_env_type(env)

        for filename in [self.env_name, self.env_type, 'rl']:
            path = os.path.join(settings.CONFIG_DIR, '{}.yml'.format(filename))
            try:
                config = self.load_file(path)
                break
            except:
                print('Failed to load {} configuration file'.format(path))
            
        self.load_dict(config)
        return path

    def load_file(self, path):
        print('Loading {} config file...'.format(path))
        with open(path, 'r') as f:
            return yaml.load(f)

    def load_dict(self, config):
        main = config['main']
        self.model_type  = main['model']
        self.policy_type = main['policy']

        if config.get(self.model_type) is None:
            self.model = self.defaults[self.model_type]
        else:
            self.model  = config[self.model_type]
        
        if config.get(self.policy_type) is None:
            self.policy = self.defaults[self.policy_type]
        else:
            self.policy = config[self.policy_type]

        self.env     = config['environment']
        self.trainer = config['main']
    
    def get_env_type(self, env_name):
        for env_type, envs in settings.envs.items():
            if env_name in envs:
                return env_type
        
    def save(self, path):
        with open(os.path.join(path, 'config.yml'), 'w') as config_file:
            yaml.dump(self.join_config(), config_file)

    def join_config(self):
        model = self.model_type
        config = {
            self.model_type: self.model,
            self.policy_type: self.policy,
            'main': self.trainer,
            'environment': self.env
        }

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
    def __init__(self, env=None):
        self.env_name = ''
        self.env_path = None

        self.instance_name = None
        self.instance_path = None

        self.trainer = Trainer()
        self.config = ConfigManager()

        if env is not None:
            self.set_env(env)

    def set_env(self, env):
        self.env_name = env
        self.env_path = os.path.join(settings.TRAINED_MODELS, env)
        os.makedirs(self.env_path, exist_ok=True)
        config_path = self.config.load(env)
        return config_path

    def new_instance(self, namestamp=None):

        # Assign a unique numerical ID to an instance
        numerical_ids = [int(x.split('_')[0]) for x in os.listdir(self.env_path)]
        try:
            unique_id = max(numerical_ids) + 1
        except:
            unique_id = 0

        # Check if some IDs are missing (e.g. deleted)
        for num in range(len(numerical_ids)):
            if num not in numerical_ids:
                unique_id = num
                break

        if namestamp is None:
            date = datetime.datetime.now().strftime("%m-%d_%H-%M")
            namestamp = "{}_{}_{}_{}_{}".format(
                self.env_name,
                self.config.model_type, 
                self.config.policy_type,
                self.config.trainer['n_workers'], 
                date)
        self.instance_name = str(unique_id) + '_' + namestamp 
        self.instance_path = os.path.join(self.env_path, self.instance_name)
        os.makedirs(self.instance_path, exist_ok=True)
        print(self.instance_path)
        self.config = ConfigManager(env_name=self.env_name)
        self.trainer.create_model(config=self.config, path=self.instance_path)

    def load_instance(self, path=None, num=None):
        """
        Loads an instance from the specified path
        """
        if num is not None:
            subdirs = []
            for f in os.listdir(self.env_path):
                fpath = os.path.join(self.env_path, f)
                print(f)
                if os.path.isdir(fpath) and num == int(f.split('_')[0]):
                    print(fpath)
                    path = fpath
                    break

        config_path = os.path.join(path, 'config.yml')
        self.config.load_dict(self.config.load_file(config_path))
        self.trainer.load_model(path=path, config=self.config)
        self.instance_path = path
        self.instance_name = os.path.split(path)[-1]
    
    def save_instance(self):
        """
        Saves the current instance (model weights and config files)
        """
        # try:
        print(self.instance_path)
        if self.config.trainer.get('steps_trained') is None:
            self.config.trainer['steps_trained'] = self.trainer.steps_trained
        else:
            self.config.trainer['steps_trained'] += self.trainer.steps_trained
        self.config.save(self.instance_path)
        self.trainer.save_model()
        # except:
        #     print('Nothing to save.')

    def tensorboard(self, browser=True):
        # Kill current session
        self._tensorboard_kill()

        # Open the dir of the current env
        cmd = 'tensorboard --logdir ' + self.instance_path
        print('Launching tensorboard at {}'.format(self.instance_path))
        DEVNULL = open(os.devnull, 'wb')
        subprocess.Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
        
        if browser:
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


