from dataclasses import dataclass, field
from typing import List

@dataclass
class EnvConfig:
    pass

@dataclass
class MainConfig:
    # environment: str = ''
    model: str = 'PPO2'
    policy: str = 'CustomMlpPolicy'
    steps_trained: int = 0
    steps_to_train: int = 1000000
    # save_every : int = 10000
    n_workers: int = 4
    highest_reward: float = -1000.
    logs: List[int] = field(default_factory=lambda: [])

@dataclass
class PPO2:
    gamma: float = 0.99  # Discount factor for future rewards
    n_steps: int = 256 # Batch size (n_steps * n_workers)
    ent_coef: float = 0.01 # Entropy loss coefficient (higher values encourage more exploration)
    learning_rate: float = 0.00025 # LR
    vf_coef: float = 0.5 # The contribution of value function loss to the total loss of the network
    max_grad_norm: float = 0.5 # Max range of the gradient clipping
    lam: float = 0.95 # Generalized advantage estimation, for controlling variance/bias tradeoff
    nminibatches: int = 4 # Number of minibatches for SGD/Adam updates
    noptepochs: int = 4 # Number of iterations for SGD/Adam
    cliprange: float = 0.2 # Clip factor for PPO (the action probability distribution of the updated policy cannot differ from the old one by this fraction [measured by KL divergence])
    verbose: int = 2

@dataclass
class DQN:
    gamma: 0.99
    learning_rate: 0.001
    buffer_size: 20000
    exploration_fraction: 0.1
    exploration_final_eps: 0.01
    train_freq: 1
    batch_size: 32
    learning_starts: 1000
    target_network_update_freq: 500
    prioritized_replay: False
    prioritized_replay_alpha: 0.2
    prioritized_replay_beta0: 0.4
    prioritized_replay_beta_iters: None
    prioritized_replay_eps: 0.000001
    param_noise: False
    verbose: 1
    _init_setup_model: True

@dataclass
class A2C:
    gamma: 0.99
    learning_rate: 0.0007
    n_steps: 5
    vf_coef: 0.25
    ent_coef: 0.01
    max_grad_norm: 0.5
    alpha: 0.99
    epsilon: 0.0001
    lr_schedule: 'constant'
    verbose: 0
#

@dataclass
class DQN:
    gamma: 0.99
    n_steps: 20
    num_procs: 1
    q_coef: 0.5
    ent_coef: 0.01
    max_grad_norm: 10
    learning_rate: 0.0007
    lr_schedule: 'linear'
    rprop_alpha: 0.99
    rprop_epsilon: 0.0001
    buffer_size: 5000
    replay_ratio: 4
    replay_start: 1000
    correction_term: 10.0
    trust_region: True
    alpha: 0.99
    delta: 1
    verbose: 0

@dataclass
class ACKTR:
    gamma: 0.99
    nprocs: 1
    n_steps: 20
    ent_coef: 0.01
    vf_coef: 0.25
    vf_fisher_coef: 1.0
    learning_rate: 0.25
    max_grad_norm: 0.5
    kfac_clip: 0.001
    lr_schedule: 'linear'
    verbose: 0
    async_eigen_decomp: False

############
### Policies

@dataclass
class CustomMlpPolicy:
    layers: List[int] = field(default_factory=lambda: [32, 16])

@dataclass
class CustomCnnPolicy:
    filters: List[int] = field(default_factory=lambda: [32, 16, 8])
    kernel_size: List[int] = field(default_factory=lambda: [5, 3, 3])
    stride: List[int] = field(default_factory=lambda: [1, 1, 1])
    layers: List[int] = field(default_factory=lambda: [32, 32]) # Number of nodes in the layers of the shared part of the fully connected network
    activ: str = 'relu'
    pd_init_scale: float = 0.05
    conv_init_scale: float = 1.4
    kernel_initializer: str = 'glorot_normal_initializer'
    init_bias: float = 0.5
