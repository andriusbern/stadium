from setuptools import setup
import time, os

packages = ['numpy==1.16.2',        
            # 'tensorflow',   # ML
            'tensorflow==1.13.1',
            'gym',
            'opencv-python',# Image processing
            'matplotlib',   # Visualization        
            'stable_baselines',
            'ipython',
            'pyyaml'
            ]
setup(
    name='rl',
    description='RL training library.',
    long_description='',
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