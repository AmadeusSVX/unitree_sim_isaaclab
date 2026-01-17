# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
G1 Bowing Motion Task

This task trains a G1 robot to perform a bowing motion:
1. Start in standing position
2. Bow forward by bending waist/hip joints
3. Return to standing position

The motion is cyclic and the robot must maintain balance throughout.
"""

import gymnasium as gym

from . import bow_g1_29dof_env_cfg
from .agents.rsl_rl_ppo_cfg import BowG129PPORunnerCfg

# Register the environment with Gymnasium
gym.register(
    id="Isaac-Bow-G129-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bow_g1_29dof_env_cfg.BowG129EnvCfg,
        "rsl_rl_cfg_entry_point": BowG129PPORunnerCfg,
    },
)

# Also register a version with fewer environments for testing
gym.register(
    id="Isaac-Bow-G129-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": bow_g1_29dof_env_cfg.BowG129EnvCfg,
        "rsl_rl_cfg_entry_point": BowG129PPORunnerCfg,
    },
)
