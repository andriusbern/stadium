from setuptools import setup
import time, os

packages = [
    # 'numpy==1.16.2',        
            # 'tensorflow==1.8.0',
            'gym[box2d]',
            'gym[classic]',
            'opencv-python',
            'stable_baselines',
            'pyyaml',
            'PyQt5',
            'pyqtgraph',
            'sklearn',
            'imageio',
            'pybullet']

setup(
    name='rl',
    description='RL training library.',
    long_description='GUI for OpenAI Gym, training/testing scripts, environment/algorithm configuration scheme.',
    version='0.9',
    packages=['rl'],
    scripts=[],
    author='Andrius Bernatavicius',
    author_email='andrius.bernatavicius@vanderlande.com',
    url='none',
    download_url='none',
    install_requires=packages
)

print ("Installation complete.\n")