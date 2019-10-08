

import os, sys

#############
# Directories

MAIN_DIR = os.path.dirname(os.path.realpath(__file__))


ENVIRONMENTS = ['OpenAI gym environments',
               '       MountainCarContinuous-v0',
               '       Pendulum-v0',
               '       BipedalWalker-v2',
               '       BipedalWalkerHardcore-v2',
               '       LunarLanderContinuous-v2',
               'Pybullet environments',
               '       AntPyBulletEnv-v0',
               '       HopperPyBulletEnv-v0',
               '       HalfCheetahPyBulletEnv-v0',
               '--Custom V-Rep environments',
               '       Quadruped',
               '       NaoBalancing',]



parameters = {}
parameters['Bin Packing'] = {    
   # Environment parameters

   'Environment': {

   },

   # Learning parameters
   'Learning': {
      'gamma'          : 0.99, # Discount factor
      'lambda'         : 0.99, # Generalize Advantage Estimation
      'epochs'         : 10,
      'batch_size'     : 200,
      'lr_actor'       : 0.0001,
      'lr_critic'      : 0.0001,
      'clipping'       : 0.2,
      'action_scaling' : 2,
      'take_mean'      : 0.8,
      'log_variance'   : -1.0,
      'random_seed'    : 0,
   },

   # Neural net parameters
   'Networks' :    {
      'network_type': 'convolutional',
      'filters': [16, 16],     # Number of convolutional layers and filters
      'strides': [2, 2],
      'kernel' : [[7, 7], [7, 7]],
      'activation_h_c' : 'relu',
      'activation_o_c' : None,
      'activation_h_a' : 'relu',
      'activation_o_a' : 'sigmoid',
   },

   # Drawing and plotting parameters
   'Rendering' : {
      'draw_images': True,
      'draw_every' : 1,
      'verbose': True,
      '              ' : ''
   }
}

parameters['MountainCarContinuous-v0'] = {    
   # Environment parameters
   'environment': {

   },

   # Learning parameters
   'Learning': {
      # 'algorithm'      : , # Add parallel ppo and ddpg in the future versions
      'gamma'          : 0.99, # Discount factor
      'lambda'         : 0.99, # Generalize Advantage Estimation
      'epochs'         : 20,
      'batch_size'     : 2,
      'lr_actor'       : 0.001,
      'lr_critic'      : 0.002,
      'clipping'       : 0.2,
      'action_scaling' : 1,
      'log_variance'   : -1.0,
      'random_seed'    : 0,
   },

   # Neural net parameters
   'Networks' :    {
      'network_type': 'fully_connected',
      'layers_actor'   : [10, 10],     # Number of convolutional layers and filters
      'layers_critic'  : [10, 10],
      'activation_h_c' : 'relu',
      'activation_o_c' : None,
      'activation_h_a' : 'tanh',
      'activation_o_a' : 'tanh',
   },

   # Drawing and plotting parameters
   'Rendering' : {
      'draw_images': True,
      'draw_every' : 1,
      'print_stats': True,
      '              ' : '',
   }
}

parameters['BipedalWalker-v2'] = {    
   # Environment parameters
   'environment': {

   },

   # Learning parameters
   'Learning': {
      # 'algorithm'      : , # Add parallel ppo and ddpg in the future versions
      'action_scaling' : 1,
      'log_variance'   : -1.0,
      'gamma'          : 0.99, # Discount factor
      'lambda'         : 0.99, # Generalize Advantage Estimation
      'epochs'         : 20,
      'batch_size'     : 10,
      'lr_actor'       : 0.001,
      'lr_critic'      : 0.001,
      'clipping'       : 0.2,
      'action_scaling' : 1,
      'log_variance'   : -1.0,
      'random_seed'    : 0,
      
   },

   # Neural net parameters
   'Networks' :    {
      'network_type': 'fully_connected',
      'layers_actor': [10, 10],     # Number of convolutional layers and filters
      'layers_critic' : [10, 10],
      'activation_h_c' : 'relu',
      'activation_o_c' : None,
      'activation_h_a' : 'tanh',
      'activation_o_a' : 'tanh',
   },

   # Drawing and plotting parameters
   'Rendering' : {
      'draw_images': True,
      'draw_every' : 1,
      'print_stats': True,
      '              ' : '',
   }
}


parameters['BipedalWalker-v2'] = {    
   # Environment parameters
   'environment': {

   },

   # Learning parameters
   'Learning': {
      # 'algorithm'      : , # Add parallel ppo and ddpg in the future versions
      'action_scaling' : 1,
      'log_variance'   : -1.0,
      'gamma'          : 0.99, # Discount factor
      'lambda'         : 0.99, # Generalize Advantage Estimation
      'epochs'         : 20,
      'batch_size'     : 10,
      'lr_actor'       : 0.002,
      'lr_critic'      : 0.005,
      'clipping'       : 0.12,
      'action_scaling' : 1,
      'log_variance'   : -1.0,
      'random_seed'    : 2,
      
   },

   # Neural net parameters
   'Networks' :    {
      'network_type': 'fully_connected',
      'layers_actor': [128, 64],     # Number of convolutional layers and filters
      'layers_critic' : [128, 64],
      'activation_h_c' : 'relu',
      'activation_o_c' : None,
      'activation_h_a' : 'tanh',
      'activation_o_a' : 'tanh',
   },

   # Drawing and plotting parameters
   'Rendering' : {
      'draw_images': True,
      'draw_every' : 1,
      'print_stats': True,
      '              ' : '',
   }
}

parameters['BipedalWalkerHardcore-v2'] = {    
   # Environment parameters
   'environment': {

   },

   # Learning parameters
   'Learning': {
      # 'algorithm'      : , # Add parallel ppo and ddpg in the future versions
      'gamma'          : 0.99, # Discount factor
      'lambda'         : 0.99, # Generalize Advantage Estimation
      'epochs'         : 20,
      'batch_size'     : 10,
      'lr_actor'       : 0.001,
      'lr_critic'      : 0.001,
      'clipping'       : 0.2,
      'random_seed'    : 0,
      'action_scaling' : 1,
      'log_variance'   : -1.0,
   },

   # Neural net parameters
   'Networks' :    {
      'network_type': 'fully_connected',
      'layers_actor': [10],     # Number of convolutional layers and filters
      'layers_critic' : [10, 10],
      'activation_h_c' : 'relu',
      'activation_o_c' : None,
      'activation_h_a' : 'tanh',
      'activation_o_a' : 'tanh',
   },

   # Drawing and plotting parameters
   'Rendering' : {
      'draw_images': True,
      'draw_every' : 1,
      'print_stats': True,
      '              ' : '',
   }
}

parameters['Pendulum-v0'] = {    
   # Environment parameters
   'environment': {

   },

   # Learning parameters
   'Learning': {
      'gamma'          : 0.99, # Discount factor
      'lambda'         : 0.99, # Generalize Advantage Estimation
      'epochs'         : 10,
      'batch_size'     : 5,
      'lr_actor'       : 0.01,
      'lr_critic'      : 0.01,
      'clipping'       : 0.2,
      'action_scaling' : 1,
      'log_variance'   : -1.0,
      'random_seed'    : 1,
   },

   # Neural net parameters
   'Networks' :    {
      'network_type': 'fully_connected',
      'layers_actor'   : [16, 16],     # Number of convolutional layers and filters
      'layers_critic'  : [16, 16],
      'activation_h_c' : 'relu',
      'activation_o_c' : None,
      'activation_h_a' : 'tanh',
      'activation_o_a' : 'tanh',
   },

   # Drawing and plotting parameters
   'Rendering' : {
      'draw_images': True,
      'draw_every' : 1,
      'print_stats': True,
      '              ' : '',
   }
}

parameters['Quadruped'] = {    
   # Environment parameters
   'environment': {

   },

   # Learning parameters
   'Learning': {
      # 'algorithm'      : , # Add parallel ppo and ddpg in the future versions
      'gamma'          : 0.99, # Discount factor
      'lambda'         : 0.99, # Generalize Advantage Estimation
      'epochs'         : 20,
      'batch_size'     : 20,
      'lr_actor'       : 0.001,
      'lr_critic'      : 0.005,
      'clipping'       : 0.15,
      'action_scaling' : 1,
      'log_variance'   : -1.0,
      'random_seed'    : 2,
   },

   # Neural net parameters
   'Networks' :    {
      'network_type': 'fully_connected',
      'layers_actor'   : [128, 128, 64],     
      'layers_critic'  : [128, 128, 64],
      'activation_h_c' : 'relu',
      'activation_o_c' : None,
      'activation_h_a' : 'tanh',
      'activation_o_a' : 'tanh',
   },

   # Drawing and plotting parameters
   'Rendering' : {
      'draw_images': True,
      'draw_every' : 1,
      'print_stats': True,
      '              ' : '',
   }
}

parameters['Default'] = {    
   # Environment parameters
   'environment': {

   },

   # Learning parameters
   'Learning': {
      # 'algorithm'      : , # Add parallel ppo and ddpg in the future versions
      'gamma'          : 0.99, # Discount factor
      'lambda'         : 0.99, # Generalize Advantage Estimation
      'epochs'         : 10,
      'batch_size'     : 5,
      'lr_actor'       : 0.005,
      'lr_critic'      : 0.005,
      'clipping'       : 0.2,
      'action_scaling' : 1,
      'log_variance'   : -1.0,
      'random_seed'    : 0,
   },

   # Neural net parameters
   'Networks' :    {
      'network_type': 'fully_connected',
      'layers_actor': [128, 128],     # Number of convolutional layers and filters
      'layers_critic' : [128, 128],
      'activation_h_c' : 'relu',
      'activation_o_c' : None,
      'activation_h_a' : 'tanh',
      'activation_o_a' : 'tanh',
   },

   # Drawing and plotting parameters
   'Rendering' : {
      'draw_images': True,
      'draw_every' : 1,
      'print_stats': True,
      '              ' : ''
   }
}

tooltips = { 
   # Learning parameters
   
      'algorithm'      : 'The algorithm to use. Default: Proximal policy optimization (PPO).', # Add parallel ppo and ddpg in the future versions
      'gamma'          : 'Discount factor of the future rewards. Values in range [0-1], where high values of this parameter will make the agent focus more on future rewards. Typical range [0.95 - 0.995] depending on how sparse the rewards are. Good value for most problems - 0.99.',
      'lambda'         : 'Coefficient for the Generalized Advantage Estimation (GAE). Smooths out the discounted reward and ensures more stable training. Keep at 0.99 in most cases.',
      'epochs'         : 'Number of epochs for stochastic gradient descent during network updates. Values in range [5-30] (higher values will make the training faster and but more unstable).',
      'batch_size'     : 'Number of episodes to use for one network update. Value should depend on the average length of episodes. A good range of steps for each batch is in the range [500-5000], depending on the complexity of the state and action spaces. Simple environments train better with small batch sizes.',
      'lr_actor'       : 'Learning rate of the actor (policy) network. Values in range [0.01 - 0.0001], use higher learning rates for simple problems.',
      'lr_critic'      : 'Learning rate of the critic (value estimation) network. Values in range [0.01 - 0.0001], use higher learning rates for simple problems, and this can be slightly higher than the actor learning rate.',
      'clipping'       : 'The clipping parameter for PPO algorithm (prevents the new policy deviating too far from current policy after an update). Values typically in range of [0.1-0.3], where a lower value will provide slower but more stable training process.',
      'action_scaling' : 'Control the variance of the distribution from which actions are drawn. Keep this value unchanged in most cases.',
      'rewards'        : 'Reward functions to use. ',
      'log_variance'   : 'The log of the variance of the distribution from which the actions are sampled. Keep this value unchanged.',

   # Neural net parameters
   
      'network_type': 'The type of neural network architecture for actor and critic networks. For these problems you only need fully connected layers.',
      'filters': 'Number of convolutional layers and the number of filters used in them.',     # Number of convolutional layers and filters
      'strides': 'Stride to use during convolutions.',
      'kernel' : 'Size of the kernel to use for each convolutional layer.',
      'layers_actor' : 'Number of layers and hidden units in them of the actor network. E.g. first hidden layer with 16 units and second hidden layer with 8 units = [16, 8].',
      'layers_critic': 'Number of layers and hidden units in them of the critic network. E.g. first hidden layer with 16 units and second hidden layer with 8 units = [16, 8].',
      'activation_h_c' : 'Activation of the hidden layers of the critic network. \nOptions: \n  None - linear,\n  relu - rectified linear unit [0-inf], \n  sigmoid - logistic function [0-1],\n  tanh - hyperbolic tangent.',
      'activation_o_c' : 'Activation of the output layer of the critic network. \nOptions: \n  None - linear,\n  relu - rectified linear unit [0-inf], \n  sigmoid - logistic function [0-1],\n  tanh - hyperbolic tangent [-1:1].',
      'activation_h_a' : 'Activation of the hidden layers of the actor network. \nOptions: \n  None - linear,\n  relu - rectified linear unit [0-inf], \n  sigmoid - logistic function [0-1],\n  tanh - hyperbolic tangent [-1:1].',
      'activation_o_a' : 'Activation of the output layer of the actor network. \nOptions: \n  None - linear,\n  relu - rectified linear unit [0-inf], \n  sigmoid - logistic function [0-1],\n  tanh - hyperbolic tangent [-1:1].',
      '              ' : '',
      'random_seed'    : 'The random seed of tensorflow weight initializer. No random seed - 0.',
      'print_stats'    : 'Output the episode statistics to the terminal.',
      # Drawing and plotting parameters
   
      'draw_images': 'Enable rendering.',
      'draw_every' : 'Draw the last frame of every n-th episode.',
      'verbose': 'Print the episode summary.',
   }
