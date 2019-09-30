import rl.environments
from rl.baselines import Trainer, get_parameters
import os, argparse
"""
A script for training a RL model in a specified environment
A configuration file from ../config/* that corresponds to the name of your environment or the 
environment type.

Usage: python train.py --env_name TestEnv --subdir TestSubdirectory --name NewModel
    or 
       python train.py -e TestEnv -s TestSubdirectory -n NewModel -m DQN
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment', type=str, help='Name of the environment. Can be either a any gym environment or a custom one defined in rl.environments')
    parser.add_argument('-s', '--subdir', type=str, help='Subdirectory where the trained model is going to be stored (useful for separating tensorboard logs): e.g. -> ../trained_models/env_type/env/[SUBDIR]/0_model/*')
    parser.add_argument('-n', '--name', type=str, default=None, help='Unique identifier of the model, e.g. -> ../trained_models/env_type/env/subdir/0_[NAME]/*')
    parser.add_argument('-m', '--model', type=str, default=None, help='Reinforcement learning model to use. PPO / ACER / ACKTR / DQN / .')
    parser.print_help()
    args = parser.parse_args()

    trainer = Trainer(args.environment, args.subdir)
    config = get_parameters(args.environment)
    if args.model is not None:
        config['main']['model'] = args.model
    trainer.create_model(name=args.name, config_file=config)
    trainer._tensorboard()
    trainer.train()