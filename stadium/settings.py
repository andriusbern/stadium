import os, sys

class GlobalConfig():
    """
    Singleton class for global configuration
    """

    ####
    # Config management
    if getattr(sys, 'frozen', False):
        SRC_DIR = sys._MEIPASS
    else:
        SRC_DIR = os.path.dirname(__file__)
    TRAINED_MODELS = os.path.join(SRC_DIR, 'trained_models')
    CONFIG_DIR = os.path.join(SRC_DIR, 'config')
    DEFAULTS = os.path.join(CONFIG_DIR, 'defaults')
    UTILS = os.path.join(SRC_DIR, 'utils')
    ICONS = os.path.join(UTILS, 'icons')

    ENV = ''
    ENV_DIR = ''
    INSTANCE_DIR = None
    INSTANCE_NAME = None
    CONF_PATH = None

    STEPS = 1024 #8192

    config = {}
    defaults = {}
    env_list = [
        'Pendulum-v0',
        'MountainCarContinuous-v0',
        'BipedalWalker-v2',
        'BipedalWalkerHardcore-v2',
        'Snake',
        'Sokoban-v0',
        # 'procgen:procgen-coinrun-v0',
        # 'procgen:procgen-miner-v0',
        # 'procgen:procgen-ninja-v0',
        # 'procgen:procgen-plunder-v0',
        # 'procgen:procgen-starpilot-v0',
        # 'procgen:procgen-leaper-v0',
        # 'procgen:procgen-jumper-v0',
        # 'procgen:procgen-maze-v0',
        # 'procgen:procgen-heist-v0',
        # 'procgen:procgen-fruitbot-v0',
        # 'procgen:procgen-dodgeball-v0',
        # 'procgen:procgen-climber-v0',
        # 'procgen:procgen-chaser-v0',
        # 'procgen:procgen-caveflyer-v0',
        # 'procgen:procgen-bossfight-v0',
        # 'procgen:procgen-bigfish-v0',
        'Assault-v4',
        'Atlantis-v4',
        'Alien-v4',
        'BattleZone-v4',
        'Boxing-v4',
        'Breakout-v4',
        'Kangaroo-v4',
        'MontezumaRevenge-v4',
        'Skiing-v4',
        'UpNDown-v4',
    ]

    translated_envs = {
        'Snake' : 'Snake',
        'Pendulum-v0': 'Pendulum',
        'BipedalWalker-v2':'BipedalWalker',
        'BipedalWalkerHardcore-v2': 'BipedalWalkerHard',
        'MountainCarContinuous-v0': 'MountainCar',
        'Sokoban-v0': 'Sokoban',
        'Sokoban-large-v0':'Sokoban-large-v0',
        'Sokoban-large-v1':'Sokoban-large-v1',
        'Sokoban-large-v2':'Sokoban-large-v2',
        'Sokoban-huge-v0':'Sokoban-huge-v0',
        'Assault-v4': 'Assault',
        'Atlantis-v4': 'Atlantis',
        'Alien-v4': 'Alien',
        'BattleZone-v4': 'Battlezone',
        'Boxing-v4': 'Boxing',
        'Breakout-v4': 'Breakout',
        'Kangaroo-v4': 'Kangaroo',
        'MontezumaRevenge-v4': 'MontezumaRevenge',
        'Skiing-v4': 'Skiing',
        'UpNDown-v4': 'UpNDown',
        'procgen:procgen-coinrun-v0': 'Coinrun',
        'procgen:procgen-miner-v0': 'Miner',
        'procgen:procgen-ninja-v0': 'Ninja',
        'procgen:procgen-plunder-v0': 'Plunder',
        'procgen:procgen-starpilot-v0': 'Starpilot',
        'procgen:procgen-leaper-v0': 'Leaper',
        'procgen:procgen-jumper-v0': 'Jumper',
        'procgen:procgen-maze-v0': 'Maze',
        'procgen:procgen-heist-v0': 'Heist',
        'procgen:procgen-fruitbot-v0': 'Fruitbot',
        'procgen:procgen-dodgeball-v0': 'Dodgeball',
        'procgen:procgen-climber-v0':'Climber',
        'procgen:procgen-chaser-v0':'Chaser',
        'procgen:procgen-caveflyer-v0':'Caveflyer',
        'procgen:procgen-bossfight-v0':'Bossfight',
        'procgen:procgen-bigfish-v0':'Bigfish'
    }

    env_types = {
        'procgen':[
            'procgen:procgen-coinrun-v0',
            'procgen:procgen-miner-v0',
            'procgen:procgen-ninja-v0',
            'procgen:procgen-plunder-v0',
            'procgen:procgen-starpilot-v0',
            'procgen:procgen-leaper-v0',
            'procgen:procgen-jumper-v0',
            'procgen:procgen-maze-v0',
            'procgen:procgen-heist-v0',
            'procgen:procgen-fruitbot-v0',
            'procgen:procgen-dodgeball-v0',
            'procgen:procgen-climber-v0',
            'procgen:procgen-chaser-v0',
            'procgen:procgen-caveflyer-v0',
            'procgen:procgen-bossfight-v0',
            'procgen:procgen-bigfish-v0']
    }

    inv_translated_envs = {v: k for k, v in translated_envs.items()}

    translated_parameters = dict(
        # RL parameters
        gamma='Discount factor',
        n_steps='Steps per epoch',
        ent_coef='Entropy coefficient',
        learning_rate='Learning rate',
        vf_coef='Value function coefficient',
        max_grad_norm='Max gradient normalization',
        lam='Lambda',
        nminibatches='# Minibatches',
        noptepochs='Epochs',
        cliprange='Clipping range',
        #='Full TB log',
        verbose='Verbosity',
        highest_reward='Highest reward',

        steps_to_train='# of Steps to train for',
        steps_trained='Steps trained',
        n_workers='# Workers',
        save_every='Save every n steps',
        model='Model',
        policy='NN type',
        normalize='Normalize inputs',
        )
    inv_translated_parameters = {v: k for k, v in translated_parameters.items()}
