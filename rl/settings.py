import os, sys

OS = sys.platform

#############
# Directories
MAIN_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(MAIN_DIR, os.pardir))
TRAINED_MODELS = os.path.join(MAIN_DIR, 'trained_models')
CONFIG   = os.path.join(MAIN_DIR, 'config')

