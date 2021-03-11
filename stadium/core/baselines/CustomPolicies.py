from stable_baselines.common.policies import ActorCriticPolicy, register_policy, FeedForwardPolicy, LstmPolicy, RecurrentActorCriticPolicy
import tensorflow as tf
from stable_baselines.deepq.policies import FeedForwardPolicy as DQNffwd
from stable_baselines.common.policies import conv, linear, lstm
import numpy as np
import copy


def custom_cnn(scaled_images, config, **kwargs):
    """
    Custom CNN Architecture builder from .yml configuration files

    Arguments to this function are passed from 
    rl/config/env_name.yml  -> policies: CustomCnnPolicy
    """
    try:
        activ = getattr(tf.nn, config.activ)
    except Exception as e:
        print(e, 'Invalid activation function.')

    init_scale = config.conv_init_scale
    
    # First layer
    out = activ(conv(
        input_tensor=scaled_images, 
        scope='c0', 
        n_filters=config.filters[0], 
        filter_size=config.kernel_size[0], 
        stride=config.stride[0], 
        init_scale=init_scale, 
        **kwargs))

    # Following layers
    for i, layer in enumerate(config.filters[1:]):
        out = activ(conv(
            input_tensor=out, 
            scope='c{}'.format(i+1), 
            n_filters=layer,
            filter_size=config.kernel_size[i+1], 
            stride=config.stride[i+1], 
            init_scale=init_scale, 
            **kwargs))

    n_hidden = np.prod([v.value for v in out.get_shape()[1:]])
    out = tf.reshape(out, [-1, n_hidden])

    return out


class CustomCnnPolicy(ActorCriticPolicy):
    """
    Custom CNN policy
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 config=None, **kwargs):
        super(CustomCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)
        init_scale = config.pd_init_scale
        activ = getattr(tf.nn, config.activ)
        initializer = getattr(tf, config.kernel_initializer)

        with tf.variable_scope('model', reuse=reuse):
            extracted_features = custom_cnn(self.processed_obs, config)
            flattened = tf.layers.flatten(extracted_features)
            
            # Shared layers
            shared = flattened
            # for i, layer in enumerate(config.layers):
            #     shared = activ(tf.layers.dense(shared, layer, name='fc_shared'+str(i), kernel_initializer=initializer))

            # Policy head
            pi_h = shared
            # for i, layer in enumerate(config['h_actor']):
            #     pi_h = activ(tf.layers.dense(pi_h, layer, name='pi_fc'+str(i), kernel_initializer=initializer))
            pi_latent = pi_h

            # Value head
            vf_h = shared
            # for i, layer in enumerate(config['h_critic']):
            #     vf_h = activ(tf.layers.dense(vf_h, layer, name='vf_fc'+str(i), kernel_initializer=initializer))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            # Initialize appropriate probability distribution as the NN output based on
            # the type of action space
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=init_scale, init_bias=config.init_bias)

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

    Arguments to the constructor are passed from 
    rl/config/env_name.yml  -> policies: CustomMlpPolicy
    """
    
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, config=None, **kwargs):
        config = copy.deepcopy(config)
        net_architecture = config.layers
        # net_architecture.append(dict(pi=config.h_actor, 
        #                              vf=config.h_critic))
        print('Custom MLP architecture', net_architecture)
        super(CustomMlpPolicy, self).__init__(sess,ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                                              net_arch=net_architecture,
                                              feature_extraction="mlp")

class CustomLSTMPolicy(LstmPolicy):
    """
    A custom LSTM policy architecture initializer

    Arguments to the constructor are passed from 
    rl/config/env_name.yml  -> policies: CustomLSTMPolicy
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, config=None, **_kwargs):
        config = copy.deepcopy(config)
        net_architecture = config['shared']
        net_architecture.append(dict(pi=config['h_actor'], 
                                     vf=config['h_critic']))
        print('Custom Lstm architecture: ', net_architecture)

        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, config['n_lstm'], reuse,
                         net_arch=net_architecture, layer_norm=True, feature_extraction="mlp", **_kwargs)


class CustomDQNPolicy(DQNffwd):
    """
    A custom LSTM policy architecture initializer

    Arguments to the constructor are passed from 
    rl/config/env_name.yml  -> policies: CustomDQNolicy
    """
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layer_norm=False,
                                           feature_extraction="mlp")


class CustomCnnLnLstmPolicy(RecurrentActorCriticPolicy):
    
    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=4, reuse=False, layers=None,
                 net_arch=None, act_fun=tf.tanh, cnn_extractor=custom_cnn, layer_norm=True, feature_extraction="cnn", config=None,
                 **kwargs):
        super(CustomCnnLnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                         state_shape=(2 * n_lstm, ), reuse=reuse,
                                         scale=(feature_extraction == "cnn"))
        config = config
        init_scale = config['pd_init_scale']
        activ = getattr(tf.nn, config['activ'])
        initializer = getattr(tf, config['kernel_initializer'])
        self._kwargs_check(feature_extraction, kwargs)
        net_arch = config['shared']
        net_arch.append(dict(pi=config['h_actor'],
                             vf=config['h_critic']))

        if net_arch is None:  # Legacy mode
            if layers is None:
                layers = [64, 64]

            with tf.variable_scope("model", reuse=reuse):
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, config)
                else:
                    extracted_features = tf.layers.flatten(self.processed_obs)
                    for i, layer_size in enumerate(layers):
                        extracted_features = act_fun(linear(extracted_features, 'pi_fc' + str(i), n_hidden=layer_size,
                                                            init_scale=np.sqrt(2)))
                input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
                masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                             layer_norm=layer_norm)
                rnn_output = seq_to_batch(rnn_output)
                value_fn = linear(rnn_output, 'vf', 1)

                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

            self._value_fn = value_fn
        else:  # Use the new net_arch parameter

            with tf.variable_scope("model", reuse=reuse):
                extracted_features = cnn_extractor(self.processed_obs, config)
                
                latent = tf.layers.flatten(extracted_features)
                policy_only_layers = [] 
                value_only_layers = [] 

                lstm_layer_constructed = False
                for idx, layer in enumerate(net_arch):
                    if isinstance(layer, int):
                        layer_size = layer
                        latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
                    elif layer == "lstm":
                        if lstm_layer_constructed:
                            raise ValueError("The net_arch parameter must only contain one occurrence of 'lstm'!")
                        input_sequence = batch_to_seq(latent, self.n_env, n_steps)
                        masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                        rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                                     layer_norm=layer_norm)
                        latent = seq_to_batch(rnn_output)
                        lstm_layer_constructed = True
                    else:
                        assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                        if 'pi' in layer:
                            assert isinstance(layer['pi'],
                                              list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                            policy_only_layers = layer['pi']

                        if 'vf' in layer:
                            assert isinstance(layer['vf'],
                                              list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                            value_only_layers = layer['vf']
                        break 

                latent_policy = latent
                for idx, pi_layer_size in enumerate(policy_only_layers):
                    if pi_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the policy network.")
                    assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                    latent_policy = act_fun(
                        linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

                latent_value = latent
                for idx, vf_layer_size in enumerate(value_only_layers):
                    if vf_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the value function "
                                                  "network.")
                    assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                    latent_value = act_fun(
                        linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

                if not lstm_layer_constructed:
                    raise ValueError("The net_arch parameter must contain at least one occurrence of 'lstm'!")

                self._value_fn = linear(latent_value, 'vf', 1)
                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(latent_policy, latent_value)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
        else:
            return self.sess.run([self.action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
