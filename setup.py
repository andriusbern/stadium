from setuptools import setup

packages = [
    'pyqtgraph==0.11.1',
    'pyyaml',
    'opencv-python',
    'gym==0.15.4',
    'stable_baselines==2.7.0',
    'tensorflow==2.7.2',
    'procgen',
    'dataclasses',
    'gym_sokoban',
    'box2d'
    # 'pyglet==1.5.11'
]


setup(
    name='stadium',
    description='Interactive design of learning environments.',
    long_description='GUI for OpenAI Gym, training/testing scripts, environment/algorithm configuration scheme.',
    version='0.5',
    packages=['stadium'],
    scripts=[],
    author='Andrius Bernatavicius',
    author_email='a.bernatavicius@liacs.leidenuniv.nl',
    url='none',
    download_url='none',
    install_requires=packages
)

print ("Installation complete.\n")