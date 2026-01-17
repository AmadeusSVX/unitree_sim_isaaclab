# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Standalone G1 Bowing Motion Task

This module is independent of the main tasks package to avoid pinocchio import issues.
"""

import gymnasium as gym

from .bow_env_cfg import BowG129EnvCfg
from .agents.rsl_rl_ppo_cfg import BowG129PPORunnerCfg

# Register the environment with Gymnasium
gym.register(
    id="Isaac-Bow-G129-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BowG129EnvCfg,
        "rsl_rl_cfg_entry_point": BowG129PPORunnerCfg,
    },
)

__all__ = ["BowG129EnvCfg", "BowG129PPORunnerCfg"]
