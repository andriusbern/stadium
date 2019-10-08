
import tensorflow as tf
from Policy import Policy
from ValueFunction import NNValueFunction
from QtDisplay import QtDisplay
from Utils import Scaler

import numpy as np
import scipy.signal

import os, argparse, signal, time
import settings
from collections import deque

# Environments
import gym
# import pybulletgym
# import pybullet as p
# import nao_rl

class Trainer(QtDisplay):
    """
    This class inherits all the UI elements from QtDisplay
    Contains the PPO training algorithm
    """
    def __init__(self):
        """
        Notes:
            Create the dependencies with the settings file to make reinitialization possible        
        """
        super(Trainer, self).__init__()

        self.setWindowTitle('OpenAI gym GUI')
        self.updateTimer = None
        self.startTimer = time.time()
        self.images = []

        # Placeholders
        self.scaler = None
        self.env = None
        self.obs = None
        self.actions = None
        self.valueFunction = None
        self.policy = None
        self.trajectories = []
        self.policyLoss = None
        self.episode = None
        self.mean_reward = None
        self.sums = None
        self.mean_actions = None
        self.done = True
        self.observes, self.actions, self.rewards, self.unscaled_obs = None, None, None, None
        self.step = 0
        self.obs = 0
        self.discrete = False

    def initializeEnv(self):
        """
        Initializes the actor and critic neural networks and variables related to training
        Can be called to reinitialize the network to it's original state
        """
        # Set random seed
        self.statusBox.setText('Creating environment...')
        s = self.parameters['Learning']['random_seed'] 
        from random import seed
        
        if s != 0:
            seed(s)
            tf.random.set_random_seed(s)
             
        # Create environment
        envName = self.envSelectionDropdown.currentText().strip()
        try:
            self.env = gym.make(envName)
        except:
            import rl
            from rl.baselines import get_parameters
            config = get_parameters(envName)
            self.env = getattr(rl.environments, envName)(config=config)

        # Show screen
        try:
            self.env.render(mode="human")
        except:
            pass

        self.env.reset()
        self.done = False

        self.gamma = self.parameters['Learning']['gamma']
        self.lam = self.parameters['Learning']['lambda']
        self.policy_logvar = self.parameters['Learning']['log_variance']
        self.trajectories = []

        self.obs = self.env.observation_space.shape[0]
        try:
            self.actions = self.env.action_space.shape[0]
            self.actionWidget.setYRange(self.env.action_space.low[0]-.4, self.env.action_space.high[0]+.4)
        except:
            self.actions = self.env.action_space.n
            self.discrete = True
        
        # Create the list of deques that is used for averaging out the outputs of the actor network
        # during training of the network
        self.testAction = [deque(maxlen=5) for _ in range(self.actions)]

        self.valueFunction = NNValueFunction(self.obs, self.actions, self.parameters['Learning'], self.parameters['Networks'])
        self.policy = Policy(self.obs, self.actions, self.parameters['Learning'], self.parameters['Networks'], self.policy_logvar)
        self.policyLoss = [0]
        self.episode = 0
        self.mean_reward = []
        self.sums = 0.0
        self.mean_actions = np.zeros([self.parameters['Learning']['batch_size'], 3])
        self.scaler = Scaler(self.env.observation_space.shape[0])
        self.observes, self.rewards, self.unscaled_obs = None, None, None
        self.step = 0
        self.statusBox.setText('Created {} environment.'.format(envName))
        self.buttonStatus('initialized')

    def test(self):
        """
        The test loop that uses the current policy and shows the performance step-wise
        """
        if self.done or self.testState is None:
            self.testState = self.env.reset()
        scale, offset = self.scaler.get()
        obs = self.testState.astype(np.float32).reshape((1, -1))
        obs = (obs - offset) * scale  
        action = self.policy.sample(obs).reshape((1, -1)).astype(np.float32)
        self.testState, _, self.done, _ = self.env.step(np.squeeze(action, axis=0))

        if 'EnvSpec' in str(self.env.spec):
            
            if self.recording:
                image = self.env.render('rgb_array')
                self.images.append(image)
                print('Recording...')
            else:
                self.env.render()
        else:
            try:
                self.updateImage()
            except:
                pass

        # Update the action plot 
        action = list(np.squeeze(action, axis=0))
        for i in range(self.actions):
            self.testAction[i].append(action[i])
        self.updateActions([np.mean(self.testAction[x]) for x in range(self.actions)])

    def train(self):
        """
        The training loop for running a single episode
        All the data gathered is stored in class attributes
        The updates are performed after collecting a full batch of experiences
        """
        
        if self.updateTimer is None:
            self.updateTimer = time.time()
        obs = self.env.reset()
        
        while True:
            if obs.shape[0] != self.env.observation_space.shape[0]:
                obs = self.env.reset()
            else:
                break
        observes, actions, rewards, unscaled_obs = [], [], [], []
        done = False
        step = 0.0
        scale, offset = self.scaler.get()
        
        while not done:
            obs = obs.astype(np.float32).reshape((1, -1))
            unscaled_obs.append(obs)
            obs = (obs - offset) * scale 
            observes.append(obs)
            action = self.policy.sample(obs).reshape((1, -1)).astype(np.float32)
            actions.append(action)
            obs, reward, done, _ = self.env.step(np.squeeze(action, axis=0))
            if not isinstance(reward, float):
                reward = np.asscalar(np.asarray(reward))
            rewards.append(reward)
            step += 1e-3  
            
        # Trajectories
        # batch_step = self.episode % self.parameters['Learning']['batch_size']
        self.episode += 1
        trajectory = {'observes': np.concatenate(observes),
                        'actions': np.concatenate(actions),
                        'rewards': np.array(rewards, dtype=np.float64),
                        'unscaled_obs': np.concatenate(unscaled_obs)}
        self.trajectories.append(trajectory)
        unscaled = np.concatenate([t['unscaled_obs'] for t in self.trajectories[:(self.episode-1)%self.parameters['Learning']['batch_size']+1]])
        self.scaler.update(unscaled)

        # Reward updates
        means = np.sum(rewards) 
        self.sums += means / self.parameters['Learning']['batch_size']

        # Network updating procedure
        if self.episode > 0 and self.episode % self.parameters['Learning']['batch_size'] == 0:
            self.statusBox.setText('Updating policy network...')
            self.add_value(self.trajectories, self.valueFunction)  # Add estimated values to episodes
            self.add_disc_sum_rew(self.trajectories, self.gamma)   # Calculated discounted sum of Rs
            self.add_gae(self.trajectories, self.gamma, self.lam)  # Calculate advantage
            # concatenate all episodes into single np arrays
            observes, actions, advantages, disc_sum_rew = self.build_train_set(self.trajectories)
            loss = self.policy.update(observes, actions, advantages)  # Update policy
            self.valueFunction.fit(observes, disc_sum_rew)            # Update value function
            self.trajectories = []
            self.policyLoss.append(loss) # Update the policy loss widget

            # Reward plots
            self.mean_reward.append(self.sums)
            nSteps = np.shape(observes)[0]
            print("Updating... Batch size: {}, Steps/s: {}, Learning rates (a / c): {}, {}".format(nSteps, nSteps/(time.time() - self.updateTimer), self.parameters['Learning']['lr_actor'], self.parameters['Learning']['lr_critic']))
            self.updateReward(self.mean_reward)
            self.updateLoss(self.policyLoss)

            # Reset
            self.updateTimer = time.time()
            self.sums = 0
            self.mean_actions = np.zeros([self.parameters['Learning']['batch_size'], 3])
        
        # Draw images
        if self.parameters['Rendering']['draw_images']:
            if self.episode%self.parameters['Rendering']['draw_every'] == 0:
                if 'EnvSpec' in str(self.env.spec):
                    self.env.render()
                else:
                    try:
                        self.updateImage()
                    except:
                        pass
                        
        # Print out summary
        if self.parameters['Rendering']['print_stats']:
            print('Episode: {}, Reward: {:.1f}, Steps: {}'.format(self.episode, means, int(step*1000)))
        self.statusBox.setText('Episode {}/{}, Reward: {}'.format(self.episode%self.parameters['Learning']['batch_size']+ 1, self.parameters['Learning']['batch_size'], self.sums))

    def terminate(self):
        """
        Ends the training, closes tensorflow sessions
        """
        self.policy.close_sess()
        self.valueFunction.close_sess()

    def discount(self, x, gamma):
        """
        Calculate discounted forward sum of a sequence at each point
        """
        return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

    def add_disc_sum_rew(self, trajectories, gamma):
        """
        Adds discounted sum of rewards to all time steps of all trajectories
        """
        for trajectory in trajectories:
            if gamma < 0.999:  # don't scale for gamma ~= 1
                rewards = trajectory['rewards'] * (1 - gamma)
            else:
                rewards = trajectory['rewards']
            disc_sum_rew = self.discount(rewards, gamma)
            trajectory['disc_sum_rew'] = disc_sum_rew

    def add_value(self, trajectories, valueFunctio):
        """
        Adds estimated value to all time steps of all trajectories
        """
        for trajectory in trajectories:
            observes = trajectory['observes']
            values = self.valueFunction.predict(observes)
            trajectory['values'] = values


    def add_gae(self, trajectories, gamma, lam):
        """
        Add generalized advantage estimator.
        """
        for trajectory in trajectories:
            if gamma < 0.999:  # don't scale for gamma ~= 1
                rewards = trajectory['rewards'] * (1 - gamma)
            else:
                rewards = trajectory['rewards']
            values = trajectory['values']
            # temporal differences
            tds = rewards - values + np.append(values[1:] * gamma, 0)
            advantages = self.discount(tds, gamma * lam)
            trajectory['advantages'] = advantages

    def build_train_set(self, trajectories):
        """
        Concatenate the lists of dictionaries into arrays
        """
        observes = np.concatenate([t['observes'] for t in trajectories])
        actions = np.concatenate([t['actions'] for t in trajectories])
        disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
        advantages = np.concatenate([t['advantages'] for t in trajectories])
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        return observes, actions, advantages, disc_sum_rew


if __name__ == "__main__":

    trainer = Trainer()
    trainer.main()
