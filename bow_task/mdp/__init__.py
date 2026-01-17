# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
MDP components for bowing motion task.
"""

# Import base mdp functions
from isaaclab.envs.mdp import *  # noqa: F401, F403

# Import custom observation functions
from .observations import (
    # Legacy observation functions
    get_joint_positions,
    get_joint_velocities,
    get_base_orientation,
    get_bow_phase,
    get_base_height,
    get_base_angular_velocity,
    # New observation functions (policy.onnx compatible)
    get_leg_joint_positions,
    get_leg_joint_velocities,
    get_last_action,
    LEG_JOINT_NAMES,
)

# Import custom reward functions
from .rewards import (
    # Phase 1: Standing balance rewards
    compute_upright_reward,
    compute_base_stability_reward,
    compute_joint_default_reward,
    compute_alive_reward,
    # Phase 2: Bowing motion rewards
    compute_bow_reward,
)

# Import custom termination functions
from .terminations import (
    check_fallen,
    check_bad_orientation,
)
