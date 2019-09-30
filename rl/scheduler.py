from rl.baselines import Trainer, get_parameters
import rl.environments
import os, sys, argparse, time
import rl.settings as settings

"""
A script for scheduling multiple experiments

Takes a folder containing .yml parameter files as input
Trains a model for each separate parameter file and saves them in a corresponding experiment result folder

Usage:
    python scheduler.py -e TestEnv -p experiment1 -r experiment1_models

    CTRL+C to stop training and save the model.
"""


def run_experiment(env_name, parameter_folder , result_folder, verbose=False):
    models = []
    parameter_dir = os.path.join(settings.CONFIG, parameter_folder)
    for param_file in os.listdir(parameter_dir):
        parameters = os.path.join(parameter_dir, param_file)
        models.append(Trainer(env_name, subdir=result_folder).create_model(config_location=parameters))
    models[0]._tensorboard()

    for i, model in enumerate(models):
        sep = '\n' +'='*50 + '\n'
        print(sep, 'Training model Nr {}/{}...\n'.format(i+1, len(models)))
        t0 = time.time()
        model.train()
        t = time.time() - t0
        steps = model.config['main']['n_steps']
        print('Training time: {:2f} min, steps/s: {}'.format(t/60, float(steps)/t), sep)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env_name', type=str)
    parser.add_argument('-p', '--parameter_folder', type=str)
    parser.add_argument('-r', '--result_folder', type=str)
    args = parser.parse_args()

    run_experiment(args.env_name, args.parameter_folder, args.result_folder)