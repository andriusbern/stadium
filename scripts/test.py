import rl.environments
from rl.baselines import Trainer
from rl.config import InstanceManager
from rlif.learning import Tester
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
    # parser.add_argument('-s', '--subdir', type=str, help='Subdirectory where the trained model is going to be stored (useful for separating tensorboard logs): e.g. -> ../trained_models/env_type/env/[SUBDIR]/model_num/*')
    parser.add_argument('-n', '--num', type=int, help='Unique identifier of the model, e.g. -> ../trained_models/env_type/env/subdir/[NUM]_/*')
    # parser.add_argument('-o', '--episodes', type=int, default=200, help='Number of episodes to run.')
    # parser.add_argument('-r', '--render', action='store_true', help='Render the agents.')
    args = parser.parse_args()

    manager = InstanceManager(args.environment)
    # path = os.path.join(manager.env_path, '31_RnaSynthesizer_PPO2_CustomCnnPolicy_16_06-05_09-25')
    manager.load_instance(num=args.num)

    tester = Tester(manager.trainer)
    tester.evaluate(time_limit=300)



    # model = Trainer(args.environment, args.subdir).load_instance(args.num)
    # model.run(episodes=args.episodes, render=args.render)