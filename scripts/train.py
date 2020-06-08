import rl.environments
from rl.baselines import Trainer
from rlif.learning import Tester
from rl.config import InstanceManager
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
    parser.add_argument('-n', '--name', type=str, default=None, help='Unique identifier of the model, e.g. -> ../trained_models/env_type/env/subdir/0_[NAME]/*')
    parser.print_help()
    args = parser.parse_args()

    manager = InstanceManager(args.environment)
    manager.new_instance()
    manager.tensorboard(browser=True)
    manager.trainer.train(breakpoints=1000000)
    
    manager.save_instance()
    tester = Tester(manager.trainer)
    tester.evaluate(time_limit=60)
