
import numpy as np
import tensorflow as tf
import settings
import random, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Reduce warning messages

class Policy(object):
    """
    Proximal policy optimization
    """
    def __init__(self, obs_dim, act_dim, learning_config, network_config, clipping_range=None):

        self.learning_config = learning_config
        self.network_config = network_config
        self.beta = 1.0  
        self.eta = 0
        self.policy_logvar = learning_config['log_variance']
        self.lr_multiplier = 1.0  
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.sample_from_dist = 1
        self._build_graph()
        self._init_session()

    def _build_graph(self):
        """
        Build and initialize TensorFlow graph
        """
        self.g = tf.Graph()
        with self.g.as_default() as gg:
            s = self.learning_config['random_seed']
            if s != 0:
                tf.random.set_random_seed(s)
                
            self._placeholders()
            self._policy_nn()
            self._logprob()
            self._sample()
            self._loss_train_op()
            self.init = tf.global_variables_initializer()

    def _placeholders(self):
        """
        TF Placeholders
        """
        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, shape=(None,self.obs_dim), name='obs')
        self.reshaped = tf.reshape(self.obs_ph, [-1 ,32, 32, 1])
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages')
        self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
        self.lr_ph = tf.placeholder(tf.float32, (), 'eta')
        self.old_log_vars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_log_vars')
        self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means')

    def _policy_conv_nn(self):
        """
        Convolutional Neural network
        """
        self.lr = 1e-3
        out = tf.layers.conv2d(inputs=self.reshaped, strides=2, filters=16, kernel_size=[3, 3], activation=tf.nn.relu)
        out = tf.layers.conv2d(inputs=out,           strides=2, filters=32, kernel_size=[3, 3], activation=tf.nn.relu)
        out = tf.layers.conv2d(inputs=out,           strides=2, filters=64, kernel_size=[3, 3], activation=tf.nn.relu)
        out = tf.layers.conv2d(inputs=out,           strides=2, filters=128, kernel_size=[3, 3], activation=tf.nn.relu)
        out = tf.layers.flatten(out)
        self.means = tf.layers.dense(out, self.act_dim,tf.nn.sigmoid,
                                     kernel_initializer=tf.random_normal_initializer(
                                     stddev=np.sqrt(1 / 256)), name="means")

        logvar_speed = (10 * 256) // 48
        log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
                                   tf.constant_initializer(0.0))
        self.log_vars = tf.reduce_sum(log_vars, axis=0) + self.policy_logvar

    def _policy_nn(self):
        """ 
        Fully connected neural net for policy approximation function
        """

        layers = self.network_config['layers_actor']

        out = tf.layers.dense(self.obs_ph, layers[0], getattr(tf.nn, self.network_config['activation_h_a']),
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / self.obs_dim)), name="h1")

        for layer in layers[1:]:
            out = tf.layers.dense(out, layer, getattr(tf.nn, self.network_config['activation_h_a']),
                                kernel_initializer=tf.random_normal_initializer(
                                stddev=np.sqrt(1 / layer)))
        
        self.means = tf.layers.dense(out, self.act_dim, getattr(tf.nn, self.network_config['activation_o_a']),
                                     kernel_initializer=tf.random_normal_initializer(
                                         ), name="means")
       
        logvar_speed = (10 * layers[-1]) // 48
        log_vars = tf.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
                                   tf.constant_initializer(0.0))
        self.log_vars = tf.reduce_sum(log_vars, axis=0) + self.learning_config['log_variance']

        print('Policy network parameters -- Network configuration: (in - hidden - out) {}, lr: {:.3g}'
              .format([self.obs_dim, layers, self.act_dim], self.learning_config['lr_actor']))

    def _logprob(self):
        """
        Calculate log probabilities of a batch of observations & actions
        Calculates log probabilities using previous step's model parameters and
        new parameters being trained.
        """

        logprob = -0.5 * tf.reduce_sum(self.log_vars)
        logprob += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1)
        self.logprob = logprob

        logprob_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
        logprob_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) /
                                         tf.exp(self.old_log_vars_ph), axis=1)
        self.logprob_old = logprob_old

    def _sample(self):
        """
        Sample from distribution, given observation
        """
        p = random.random()
        if p < self.sample_from_dist:
            self.sampled_act = (self.means + 
                                tf.exp(self.log_vars / 2.0) *
                                tf.random_normal(shape=(self.act_dim,)) / self.learning_config['action_scaling'])
        else:
            self.sampled_act = self.means

    def _loss_train_op(self):
        """
        Loss function of the actor net
        """
        pg_ratio = tf.exp(self.logprob - self.logprob_old)
        clipped_pg_ratio = tf.clip_by_value(pg_ratio, 1 - self.learning_config['clipping'], 1 + self.learning_config['clipping'])
        surrogate_loss = tf.minimum(self.advantages_ph * pg_ratio,
                                    self.advantages_ph * clipped_pg_ratio)
        self.loss = -tf.reduce_mean(surrogate_loss)

        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)

    def _init_session(self):
        """
        Launch TensorFlow session and initialize variables
        """
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def sample(self, obs):
        """
        Draw sample from policy distribution
        """
        feed_dict = {self.obs_ph: obs}

        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def update(self, observes, actions, advantages):
        """ 
        Update the policy network
        Returns the average loss during the epochs
        """
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: 0,
                     self.eta_ph: self.eta,
                     self.lr_ph: self.learning_config['lr_actor'] * self.lr_multiplier}
        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars],
                                                      feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np
        loss = []
        for _ in range(self.learning_config['epochs']):
            self.sess.run(self.train_op, feed_dict)
            loss.append(self.sess.run([self.loss], feed_dict))
        return np.mean(loss)

    def close_sess(self):
        """
        Close TensorFlow session
        """
        self.sess.close()
