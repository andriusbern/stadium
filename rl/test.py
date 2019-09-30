import rl.environments
from rl.baselines import Trainer, get_parameters
import os, argparse
"""
A script for testing the policy of a trained model

Usage: python retrain.py --env_name TestEnv --subdir TestSubdirectory --num 0 --render
    or 
       python retrain.py -e TestEnv -s TestSubdirectory -n 0 -t -r
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--environment', type=str, help='Name of the environment. Can be either a any gym environment or a custom one defined in rl.environments')
    parser.add_argument('-s', '--subdir', type=str, help='Subdirectory where the trained model is going to be stored (useful for separating tensorboard logs): e.g. -> ../trained_models/env_type/env/[SUBDIR]/model_num/*')
    parser.add_argument('-n', '--num', type=int, help='Unique identifier of the model, e.g. -> ../trained_models/env_type/env/subdir/[NUM]_/*')
    parser.add_argument('-o', '--episodes', type=int, default=200, help='Number of episodes to run.')
    parser.add_argument('-r', '--render', action='store_true', help='Render the agents.')
    args = parser.parse_args()

    model = Trainer(args.environment, args.subdir).load_model(args.num)
    model.run(episodes=args.episodes, render=args.render)