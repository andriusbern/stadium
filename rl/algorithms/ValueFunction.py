
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import settings
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class NNValueFunction(object):
    """ 
    NN-based state-value function
    """
    def __init__(self, obs_dim, act_dim, learning_config, network_config):
        """
        Args:
            obs_dim: number of dimensions in observation vector (int)
        """
        self.learning_config = learning_config
        self.network_config = network_config
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.lr = None  
        self._build_graph()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)
    
    def _build_conv_graph(self):
        """
        Builds a convolutional network for value approximation
        """
        self.g = tf.Graph()
        with self.g.as_default():
            s = self.learning_config['random_seed']
            if s != 0:
                tf.random.set_random_seed(s)

            self.obs_ph = tf.placeholder(tf.float32, shape=[None, self.obs_dim], name='obs_valfunc')
            self.reshaped = tf.reshape(self.obs_ph, [-1, 32, 32, 1])
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            self.lr = 1e-2 
            out = tf.layers.conv2d(inputs=self.reshaped, strides=2, filters=16,  kernel_size=[3, 3], activation=tf.nn.relu)
            out = tf.layers.conv2d(inputs=out,           strides=2, filters=32, kernel_size=[3, 3], activation=tf.nn.relu)
            out = tf.layers.conv2d(inputs=out,           strides=2, filters=64, kernel_size=[3, 3], activation=tf.nn.relu)
            out = tf.layers.conv2d(inputs=out,           strides=2, filters=128, kernel_size=[3, 3], activation=tf.nn.relu)
            out = tf.layers.flatten(out)
            print(np.shape(out))
            out = tf.layers.dense(out, 1)

            self.out = tf.squeeze(out)
            self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))  # squared loss
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def _build_graph(self):
        """
        Construct TensorFlow graph, including loss function, init op and train op
        """
        self.g = tf.Graph()
        with self.g.as_default():
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_valfunc')
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_valfunc')
            ls = self.network_config['layers_critic']
            act = self.network_config['activation_h_c']
            out = tf.layers.dense(self.obs_ph, ls[0], getattr(tf.nn, act),
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)), name="h1")
            for layer in ls[1:]:
                out = tf.layers.dense(self.obs_ph, layer, getattr(tf.nn, act),
                                      kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.obs_dim)))
            out = tf.layers.dense(out, 1,
                                  kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / ls[-1])), name='output')
            self.out = tf.squeeze(out)
            self.loss = tf.reduce_mean(tf.square(self.out - self.val_ph))  # squared loss
            optimizer = tf.train.AdamOptimizer(self.learning_config['lr_critic'])
            self.train_op = optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()

        print('Value network parameters  -- Network configuration: (in - hidden - out) {}, lr: {:.3g}'
              .format([self.obs_dim + self.act_dim, self.network_config['layers_critic'], 1], self.learning_config['lr_critic']))
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def fit(self, x, y):
        """ Fit model to current data batch + previous data batch

        Args:
            x: features
            y: target
            logger: logger to save training loss and % explained variance
        """
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self.predict(x)  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat)/np.var(y)
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        for e in range(self.learning_config['epochs']):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                _, l = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        y_hat = self.predict(x)
        loss = np.mean(np.square(y_hat - y))         
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  

    def predict(self, x):
        """ Predict method """
        feed_dict = {self.obs_ph: x}
        y_hat = self.sess.run(self.out, feed_dict=feed_dict)

        return np.squeeze(y_hat)

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()
