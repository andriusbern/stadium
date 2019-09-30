"""
This is the global settings file for this project
It contains most of the relevant settings for:
   1. Directories and folders 
   2. Image generation and plotting
   3. XML/CSV parsing -> image generation
   4. Reinforcement learning algorithm parameters
   5. Custom RL environment settings and reward shaping coefficients
   6. Dimensionality reduction parameters (conv-autoencoder, PCA)
"""
import os, sys

#############
# Directories

MAIN_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(MAIN_DIR, os.pardir))

INPUT    = os.path.join(MAIN_DIR, 'input')
OUTPUT   = os.path.join(MAIN_DIR, 'output')
RESULT   = os.path.join(MAIN_DIR, 'result') 
TRAINED_MODELS = os.path.join(MAIN_DIR, 'trained_models')
CONFIG   = os.path.join(MAIN_DIR, 'config')

# Identifiers for unique environments
ENVIRONMENT_NAMES = ['Test', 'Taxi' 'SMART', 'PICK', 'Basic', 'RL', 'Env']
