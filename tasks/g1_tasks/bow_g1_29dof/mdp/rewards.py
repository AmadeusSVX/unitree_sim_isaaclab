# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Reward functions for bowing motion task.
"""
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation


def compute_bow_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_bow_angle: float = 0.5,
    bow_duration_steps: int = 100,
    hold_duration_steps: int = 50,
    return_duration_steps: int = 100,
    angle_reward_weight: float = 1.0,
    balance_reward_weight: float = 0.5,
    smoothness_reward_weight: float = 0.2,
) -> torch.Tensor:
    """
    Compute reward for following the bowing motion trajectory.

    The reward encourages:
    - Following the target bow angle trajectory over time
    - Maintaining balance (upright base orientation)
    - Smooth motion

    Args:
        env: The environment
        robot_cfg: Robot configuration
        target_bow_angle: Target bowing angle in radians (~0.5 = 30 degrees)
        bow_duration_steps: Steps to complete the bow
        hold_duration_steps: Steps to hold the bow position
        return_duration_steps: Steps to return to standing
        angle_reward_weight: Weight for angle tracking reward
        balance_reward_weight: Weight for balance reward
        smoothness_reward_weight: Weight for smoothness reward

    Returns:
        torch.Tensor: Reward [num_envs]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device

    # Get joint indices for bowing
    joint_names = robot.data.joint_names
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel

    # Find waist pitch joint
    waist_pitch_idx = None
    left_hip_pitch_idx = None
    right_hip_pitch_idx = None

    for i, name in enumerate(joint_names):
        if "waist_pitch" in name:
            waist_pitch_idx = i
        elif "left_hip_pitch" in name:
            left_hip_pitch_idx = i
        elif "right_hip_pitch" in name:
            right_hip_pitch_idx = i

    # Calculate current bow angle
    current_bow_angle = torch.zeros(num_envs, device=device)
    num_joints_used = 0

    if waist_pitch_idx is not None:
        current_bow_angle += joint_pos[:, waist_pitch_idx]
        num_joints_used += 1

    if left_hip_pitch_idx is not None and right_hip_pitch_idx is not None:
        hip_pitch_avg = (joint_pos[:, left_hip_pitch_idx] + joint_pos[:, right_hip_pitch_idx]) / 2
        current_bow_angle += hip_pitch_avg * 0.5  # Hip contributes less
        num_joints_used += 0.5

    if num_joints_used > 0:
        current_bow_angle /= num_joints_used

    # Calculate target angle based on phase
    total_cycle_steps = bow_duration_steps + hold_duration_steps + return_duration_steps

    if hasattr(env, 'episode_length_buf'):
        current_step = env.episode_length_buf
    else:
        current_step = torch.zeros(num_envs, device=device, dtype=torch.long)

    step_in_cycle = current_step % total_cycle_steps

    # Determine target angle based on phase
    target_angle = torch.zeros(num_envs, device=device)

    # Phase 1: Bowing down
    in_bow_phase = step_in_cycle < bow_duration_steps
    bow_progress = step_in_cycle.float() / bow_duration_steps
    target_angle = torch.where(
        in_bow_phase,
        target_bow_angle * bow_progress,
        target_angle
    )

    # Phase 2: Hold
    in_hold_phase = (step_in_cycle >= bow_duration_steps) & (step_in_cycle < bow_duration_steps + hold_duration_steps)
    target_angle = torch.where(
        in_hold_phase,
        torch.full((num_envs,), target_bow_angle, device=device),
        target_angle
    )

    # Phase 3: Return
    in_return_phase = step_in_cycle >= (bow_duration_steps + hold_duration_steps)
    return_progress = (step_in_cycle - bow_duration_steps - hold_duration_steps).float() / return_duration_steps
    return_progress = torch.clamp(return_progress, 0.0, 1.0)
    target_angle = torch.where(
        in_return_phase,
        target_bow_angle * (1.0 - return_progress),
        target_angle
    )

    # Reward 1: Angle tracking
    angle_error = torch.abs(current_bow_angle - target_angle)
    angle_reward = torch.exp(-2.0 * angle_error)

    # Reward 2: Balance
    projected_gravity = robot.data.projected_gravity_b
    upright_reward = torch.clamp(projected_gravity[:, 2] + 1.0, 0.0, 1.0)
    balance_reward = upright_reward

    # Reward 3: Smoothness
    total_joint_vel = torch.sum(torch.abs(joint_vel), dim=-1)
    smoothness_reward = torch.exp(-0.01 * total_joint_vel)

    # Combine rewards
    total_reward = (
        angle_reward_weight * angle_reward +
        balance_reward_weight * balance_reward +
        smoothness_reward_weight * smoothness_reward
    )

    total_weight = angle_reward_weight + balance_reward_weight + smoothness_reward_weight
    total_reward = total_reward / total_weight

    return total_reward


def compute_alive_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_height: float = 0.3,
) -> torch.Tensor:
    """
    Reward for staying alive (not falling).

    Returns:
        torch.Tensor: Alive reward [num_envs]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    base_height = robot.data.root_pos_w[:, 2]

    alive = base_height > min_height
    reward = alive.float()

    return reward


def compute_upright_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward for maintaining upright posture.

    Returns:
        torch.Tensor: Upright reward [num_envs]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    projected_gravity = robot.data.projected_gravity_b

    # Z component should be close to -1 when upright
    upright_reward = torch.clamp(projected_gravity[:, 2] + 1.0, 0.0, 1.0)

    return upright_reward
