# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Observation functions for bowing motion task.
"""
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation


def get_joint_positions(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Get normalized joint positions.

    Returns:
        torch.Tensor: Joint positions [num_envs, num_joints]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos

    # Normalize by joint limits
    joint_pos_lower = robot.data.soft_joint_pos_limits[..., 0]
    joint_pos_upper = robot.data.soft_joint_pos_limits[..., 1]
    joint_pos_range = joint_pos_upper - joint_pos_lower
    joint_pos_range = torch.clamp(joint_pos_range, min=1e-6)

    normalized_pos = 2.0 * (joint_pos - joint_pos_lower) / joint_pos_range - 1.0

    return normalized_pos


def get_joint_velocities(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 0.1,
) -> torch.Tensor:
    """
    Get scaled joint velocities.

    Returns:
        torch.Tensor: Joint velocities [num_envs, num_joints]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    joint_vel = robot.data.joint_vel * scale
    return joint_vel


def get_base_orientation(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Get base orientation as projected gravity vector.
    This indicates how tilted the robot is from vertical.

    Returns:
        torch.Tensor: Projected gravity [num_envs, 3]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    projected_gravity = robot.data.projected_gravity_b
    return projected_gravity


def get_bow_phase(
    env: ManagerBasedRLEnv,
    bow_duration_steps: int = 100,
    hold_duration_steps: int = 50,
    return_duration_steps: int = 100,
) -> torch.Tensor:
    """
    Get the current phase of the bowing motion as sine/cosine encoding.
    This helps the policy know where it is in the bowing cycle.

    Returns:
        torch.Tensor: Phase encoding [num_envs, 2] (sin, cos)
    """
    total_cycle_steps = bow_duration_steps + hold_duration_steps + return_duration_steps

    # Get current step in episode
    if hasattr(env, 'episode_length_buf'):
        current_step = env.episode_length_buf
    else:
        current_step = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    # Calculate phase within cycle [0, 2*pi]
    step_in_cycle = current_step % total_cycle_steps
    phase = 2.0 * torch.pi * step_in_cycle.float() / total_cycle_steps

    # Encode as sin/cos for continuity
    phase_encoding = torch.stack([
        torch.sin(phase),
        torch.cos(phase)
    ], dim=-1)

    return phase_encoding


def get_base_height(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Get base height above ground.

    Returns:
        torch.Tensor: Base height [num_envs, 1]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    base_height = robot.data.root_pos_w[:, 2:3]
    return base_height


def get_base_angular_velocity(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 0.25,
) -> torch.Tensor:
    """
    Get base angular velocity in body frame.

    Returns:
        torch.Tensor: Angular velocity [num_envs, 3]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ang_vel = robot.data.root_ang_vel_b * scale
    return ang_vel
