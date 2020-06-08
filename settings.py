import os, sys, yaml, datetime
import numpy as np

class GlobalConfig():
    """
    Singleton class for global configuration
    """

    ####
    # Config management

    # Directories
    SRC_DIR = os.path.dirname(os.path.realpath(__file__))
    MAIN_DIR = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
    TRAINED_MODELS = os.path.join(SRC_DIR, 'trained_models')
    CONFIG_DIR = os.path.join(SRC_DIR, 'config')
    DEFAULTS = os.path.join(CONFIG_DIR, 'defaults')
    ICONS = os.path.join(MAIN_DIR, 'misc', 'icons')

    ENV = ''
    ENV_DIR = ''
    INSTANCE_DIR = None
    INSTANCE_NAME = None
    CONF_PATH = None

    STEPS = 16384 #8192

    config = {}
    defaults = {}

    envs = {
        'grid':  ['Snake', 'TestEnv', 'Snake2'],

        'other': ['Binpacking'],

        'vrep':
            ['Quadruped',
            'NaoTracking',
            'NaoBalancing2',
            'NaoBalancing'],

        'atari':
            ['Assault-ram-v0',
            'Atlantis-ram-v0',
            'Alien-ram-v0',
            'BattleZone-ram-v0',
            'Boxing-ram-v0',
            'Breakout-ram-v0',
            'Kangaroo-ram-v0',
            'MontezumaRevenge-ram-v0',
            'Skiing-ram-v0',
            'UpNDown-ram-v0',
            'Zaxxon-ram-v0'],

        'gym':
            ['MountainCarContinuous-v0',
            'Pendulum-v0',
            'BipedalWalker-v1',
            'BipedalWalkerHardcore-v1',
            'LunarLanderContinuous-v2'],

        'vizdoom':
            ['VizdoomBasic-v0',
            'VizdoomCorridor-v0',
            'VizdoomDefendCenter-v0',
            'VizdoomDefendLine-v0',
            'VizdoomHealthGathering-v0',
            'VizdoomMyWayHome-v0',
            'VizdoomPredictPosition-v0',
            'VizdoomTakeCover-v0',
            'VizdoomDeathmatch-v0',
            'VizdoomHealthGatheringSupreme-v0',
            ],
            
        'rlif':
            ['RnaDesign',
            'RnaImproved',
            'RnaFixer',
            'RnaSynthesizer'],
    }












    @staticmethod
    def translate(parameter):
        translate = dict(
            # RL parameters
            gamma='Gamma',
            n_steps='Steps/epoch/w',
            ent_coef='Entropy coeff',
            learning_rate='LR',
            vf_coeff='Value fn coeff',
            max_grad_norm='Max grad norm',
            lam='Lambda',
            nminibatches= '# Minibatches',
            noptepochs='Epochs',
            cliprange='Clipping range',
            full_tensorboard_log='Full TB log',
            verbose='Verbosity',

            n_workers='# Workers',
            save_every='Save every n steps',
            model='Model',
            policy='NN type',
            normalize='Normalize inputs',
            )

        return translate[parameter]       
        
        
         # assert os.path.isdir(self._env_path), 'Path {} does not exist.'.format(self._env_path)

        # folder_list = glob.glob(self._env_path + '/*') 
        # if latest:
        #     model_path = max(folder_list, key=os.path.getctime)
        # else:
        #     for folder in folder_list:
        #         print(folder)
        #         if int(folder.split('\\')[-1].split('_')[0]) == num:
        #             model_path = folder
        #             if not os.path.isfile(os.path.join(model_path, 'model.pkl')):
        #                 model_path = model_path[:-1] + '1'
        #             print('Model path:', model_path)
        #             break 