# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
MDP components for bowing motion task.
"""

# Import base mdp functions
from isaaclab.envs.mdp import *  # noqa: F401, F403

# Import custom observation functions
from .observations import (
    get_joint_positions,
    get_joint_velocities,
    get_base_orientation,
    get_bow_phase,
    get_base_height,
    get_base_angular_velocity,
)

# Import custom reward functions
from .rewards import (
    compute_bow_reward,
    compute_alive_reward,
    compute_upright_reward,
)

# Import custom termination functions
from .terminations import (
    check_fallen,
    check_bad_orientation,
)
