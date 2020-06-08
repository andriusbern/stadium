import rl.environments
from rl.baselines import Trainer
import argparse

"""
A script for continuing the training of a model

Usage: python retrain.py --env_name TestEnv --subdir TestSubdirectory --num 0
    or 
       python retrain.py -e TestEnv -s TestSubdirectory -n 0 -t
"""

if __name__ == "__main__":
    print('Retraining model... Add -h flag for displaying the help for command line arguments.')
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env_name', type=str, help='Name of the environment. Can be either a any gym environment or a custom one defined in rl.environments')
    parser.add_argument('-s', '--subdir', type=str, help='Subdirectory where the model is stored: e.g. -> ../trained_models/env_type/env/[SUBDIR]/model_num/*')
    parser.add_argument('-n', '--num', type=int, help='Unique identifier of the model, e.g. -> ../trained_models/env_type/env/subdir/[NUM]_/*')
    parser.add_argument('-t', '--tensorboard', action='store_true', help='Launch tensorboard in the current subdirectory.')
    args = parser.parse_args()

    model = Trainer(args.env_name, args.subdir).load_model(args.num)
    if args.tensorboard:
        model._tensorboard()
    model.train()
    