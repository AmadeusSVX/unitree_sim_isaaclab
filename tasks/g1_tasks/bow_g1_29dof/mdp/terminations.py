# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Termination functions for bowing motion task.
"""
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation


def check_fallen(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_base_height: float = 0.3,
) -> torch.Tensor:
    """
    Check if the robot has fallen.

    The robot is considered fallen if:
    - Base height is below minimum threshold

    Args:
        env: The environment
        robot_cfg: Robot configuration
        min_base_height: Minimum acceptable base height

    Returns:
        torch.Tensor: Boolean tensor [num_envs] - True if fallen
    """
    robot: Articulation = env.scene[robot_cfg.name]

    # Check base height
    base_height = robot.data.root_pos_w[:, 2]
    fallen = base_height < min_base_height

    return fallen


def check_bad_orientation(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_tilt: float = 1.0,  # radians (~57 degrees)
) -> torch.Tensor:
    """
    Check if the robot has tilted too much.

    Args:
        env: The environment
        robot_cfg: Robot configuration
        max_tilt: Maximum allowed tilt from vertical

    Returns:
        torch.Tensor: Boolean tensor [num_envs] - True if tilted too much
    """
    robot: Articulation = env.scene[robot_cfg.name]

    # Projected gravity indicates tilt
    # When upright, z component is close to -1
    # When tilted, z component increases toward 0
    projected_gravity = robot.data.projected_gravity_b

    # Calculate tilt angle from vertical
    # cos(tilt) = -projected_gravity_z when upright
    tilt_cos = -projected_gravity[:, 2]
    tilt_cos = torch.clamp(tilt_cos, -1.0, 1.0)

    # Check if tilt exceeds maximum
    bad_orientation = tilt_cos < torch.cos(torch.tensor(max_tilt, device=env.device))

    return bad_orientation
