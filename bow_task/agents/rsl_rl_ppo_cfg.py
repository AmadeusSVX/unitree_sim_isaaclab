# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
RSL-RL PPO agent configuration for G1 bowing motion task.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BowG129PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    PPO Runner configuration for G1 standing balance task.

    Observation space: 42 dims (ang_vel, gravity, leg_pos, leg_vel, last_action)
    Action space: 12 dims (12 leg joints)

    This configuration is optimized for learning standing balance
    with potential extension to bowing motion.
    """

    # Training parameters
    num_steps_per_env = 24  # Steps per environment before update
    max_iterations = 1500    # Total training iterations
    save_interval = 100      # Save checkpoint every N iterations

    # Experiment naming
    experiment_name = "g1_bow"
    run_name = ""
    logger = "tensorboard"

    # Normalization
    empirical_normalization = True  # Normalize observations

    # Resume training
    resume = False
    load_run = ""
    load_checkpoint = ""

    # Policy network configuration (smaller network for 42-dim obs, 12-dim action)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # Initial action noise
        actor_hidden_dims=[128, 128, 128],  # Actor network layers
        critic_hidden_dims=[128, 128, 128],  # Critic network layers
        activation="elu",  # Activation function
    )

    # PPO algorithm configuration
    algorithm = RslRlPpoAlgorithmCfg(
        # Value function
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,

        # Entropy bonus for exploration
        entropy_coef=0.01,  # Small entropy bonus for exploration

        # Learning parameters
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",  # Adaptive learning rate

        # GAE parameters
        gamma=0.99,   # Discount factor
        lam=0.95,     # GAE lambda

        # Gradient clipping
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
