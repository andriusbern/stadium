from stable_baselines.common.policies import ActorCriticPolicy, register_policy, FeedForwardPolicy, LstmPolicy
import tensorflow as tf
from stable_baselines.deepq.policies import FeedForwardPolicy as DQNffwd
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
import numpy as np
import copy


def custom_cnn(scaled_images, params, **kwargs):
    """
    Custom CNN Architecture builder for this project
    """
    try:
        activ = getattr(tf.nn, params['activ'])
    except Exception as e:
        print(e, 'Invalid activation function.')

    init_scale = params['conv_init_scale']
    # First layer
    out = activ(conv(
        input_tensor=scaled_images, 
        scope='c0', 
        n_filters=params['filters'][0], 
        filter_size=params['kernel_size'][0], 
        stride=params['stride'][0], 
        init_scale=init_scale, 
        **kwargs))

    # Following layers
    for i, layer in enumerate(params['filters'][1:]):
        out = activ(conv(input_tensor=out, 
        scope='c{}'.format(i+1), 
        n_filters=layer,
        filter_size=params['kernel_size'][i+1], 
        stride=params['stride'][i+1], 
        init_scale=init_scale, 
        **kwargs))

    n_hidden = np.prod([v.value for v in out.get_shape()[1:]])
    out = tf.reshape(out, [-1, n_hidden])

    return out

class CustomCnnPolicy(ActorCriticPolicy):
    """
    Custom CNN policy, requires a params dictionary (ParameterContainer) as an argument
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 params=None, **kwargs):
        super(CustomCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)
        print(params)
        init_scale = params['pd_init_scale']
        activ = getattr(tf.nn, params['activ'])
        initializer = getattr(tf, params['kernel_initializer'])

        with tf.variable_scope('model', reuse=reuse):
            extracted_features = custom_cnn(self.processed_obs, params)
            flattened = tf.layers.flatten(extracted_features)
            
            # Shared layers
            shared = flattened
            for i, layer in enumerate(params['shared']):
                shared = activ(tf.layers.dense(shared, layer, name='fc_shared'+str(i), kernel_initializer=initializer))

            # Policy head
            pi_h = shared
            for i, layer in enumerate(params['h_actor']):
                pi_h = activ(tf.layers.dense(pi_h, layer, name='pi_fc'+str(i), kernel_initializer=initializer))
            pi_latent = pi_h

            # Value head
            vf_h = shared
            for i, layer in enumerate(params['h_critic']):
                vf_h = activ(tf.layers.dense(vf_h, layer, name='vf_fc'+str(i), kernel_initializer=initializer))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=init_scale, init_bias=params['init_bias'])

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class CustomMlpPolicy(FeedForwardPolicy):
    """
    A custom MLP policy architecture initializer
    """
    
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, params=None, **kwargs):
        config = copy.deepcopy(params)
        net_architecture = config['shared']
        net_architecture.append(dict(pi=config['h_actor'], 
                                     vf=config['h_critic']))
        print('Custom MLP architecture', net_architecture)
        super(CustomMlpPolicy, self).__init__(sess,ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                                              net_arch=net_architecture,
                                              feature_extraction="mlp")

class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, params=None, **_kwargs):
        config = copy.deepcopy(params)
        net_architecture = config['shared']
        net_architecture.append(dict(pi=config['h_actor'], 
                                     vf=config['h_critic']))
        print('Custom Lstm architecture: ', net_architecture)

        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, config['n_lstm'], reuse,
                         net_arch=net_architecture, layer_norm=True, feature_extraction="mlp", **_kwargs)


# Custom MLP policy of two layers of size 32 each
class CustomDQNPolicy(DQNffwd):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layer_norm=False,
                                           feature_extraction="mlp")
