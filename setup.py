from setuptools import setup
import time, os

packages = ['numpy==1.16.2',        
            'tensorflow==1.15.2',
            'gym[all]',
            'opencv-python',
            'stable_baselines',
            'pyyaml',
            'PyQt5',
            'pyqtgraph']

setup(
    name='rl',
    description='RL training library.',
    long_description='GUI for OpenAI Gym, scripts, configurations',
    version='0.2',
    packages=['rl'],
    scripts=[],
    author='Andrius Bernatavicius',
    author_email='andrius.bernatavicius@vanderlande.com',
    url='none',
    download_url='none',
    install_requires=packages
)

print ("Installation complete.\n")