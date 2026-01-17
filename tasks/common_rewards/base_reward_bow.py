# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Reward function for bowing motion task.

The robot should:
1. Start in standing position
2. Bow forward by bending waist/hip joints
3. Return to standing position

This implements a cyclic bowing motion with phase tracking.
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation


def compute_bow_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # Target bow angle (radians) - positive = forward bend
    target_bow_angle: float = 0.5,  # ~30 degrees
    # Timing parameters
    bow_duration_steps: int = 100,  # Steps to complete bow
    hold_duration_steps: int = 50,   # Steps to hold bow position
    return_duration_steps: int = 100,  # Steps to return to standing
    # Reward weights
    angle_reward_weight: float = 1.0,
    balance_reward_weight: float = 0.5,
    smoothness_reward_weight: float = 0.2,
    # Joint names for bowing
    waist_pitch_joint: str = "waist_pitch_joint",
    left_hip_pitch_joint: str = "left_hip_pitch_joint",
    right_hip_pitch_joint: str = "right_hip_pitch_joint",
) -> torch.Tensor:
    """
    Compute reward for bowing motion.

    The reward encourages:
    - Following the target bow angle trajectory
    - Maintaining balance (upright base orientation)
    - Smooth motion (low joint velocity changes)

    Returns:
        torch.Tensor: Reward values for each environment [num_envs]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device

    # Get joint indices
    joint_names = robot.data.joint_names

    # Find waist pitch joint index
    waist_pitch_idx = None
    left_hip_pitch_idx = None
    right_hip_pitch_idx = None

    for i, name in enumerate(joint_names):
        if waist_pitch_joint in name:
            waist_pitch_idx = i
        elif left_hip_pitch_joint in name:
            left_hip_pitch_idx = i
        elif right_hip_pitch_joint in name:
            right_hip_pitch_idx = i

    # Get current joint positions
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel

    # Calculate current bow angle (average of waist and hip pitches if available)
    current_bow_angle = torch.zeros(num_envs, device=device)
    num_joints_used = 0

    if waist_pitch_idx is not None:
        current_bow_angle += joint_pos[:, waist_pitch_idx]
        num_joints_used += 1

    if left_hip_pitch_idx is not None and right_hip_pitch_idx is not None:
        hip_pitch_avg = (joint_pos[:, left_hip_pitch_idx] + joint_pos[:, right_hip_pitch_idx]) / 2
        current_bow_angle += hip_pitch_avg
        num_joints_used += 1

    if num_joints_used > 0:
        current_bow_angle /= num_joints_used

    # Calculate phase based on episode step count
    # Use episode length info if available, otherwise use a simple counter
    total_cycle_steps = bow_duration_steps + hold_duration_steps + return_duration_steps

    # Get current step in episode (use common counter)
    if hasattr(env, 'episode_length_buf'):
        current_step = env.episode_length_buf
    else:
        current_step = torch.zeros(num_envs, device=device, dtype=torch.long)

    # Calculate phase within the cycle
    step_in_cycle = current_step % total_cycle_steps

    # Determine target angle based on phase
    target_angle = torch.zeros(num_envs, device=device)

    # Phase 1: Bowing down (0 to bow_duration_steps)
    in_bow_phase = step_in_cycle < bow_duration_steps
    bow_progress = step_in_cycle.float() / bow_duration_steps
    target_angle = torch.where(
        in_bow_phase,
        target_bow_angle * bow_progress,
        target_angle
    )

    # Phase 2: Hold bow position (bow_duration_steps to bow_duration_steps + hold_duration_steps)
    in_hold_phase = (step_in_cycle >= bow_duration_steps) & (step_in_cycle < bow_duration_steps + hold_duration_steps)
    target_angle = torch.where(
        in_hold_phase,
        torch.full((num_envs,), target_bow_angle, device=device),
        target_angle
    )

    # Phase 3: Return to standing (bow_duration_steps + hold_duration_steps to total_cycle_steps)
    in_return_phase = step_in_cycle >= (bow_duration_steps + hold_duration_steps)
    return_progress = (step_in_cycle - bow_duration_steps - hold_duration_steps).float() / return_duration_steps
    return_progress = torch.clamp(return_progress, 0.0, 1.0)
    target_angle = torch.where(
        in_return_phase,
        target_bow_angle * (1.0 - return_progress),
        target_angle
    )

    # Reward 1: Angle tracking reward (how close to target angle)
    angle_error = torch.abs(current_bow_angle - target_angle)
    angle_reward = torch.exp(-2.0 * angle_error)  # Exponential reward, max 1.0 at perfect tracking

    # Reward 2: Balance reward (penalize tilting/falling)
    # Use projected gravity to detect if robot is upright
    projected_gravity = robot.data.projected_gravity_b
    # Z component should be close to -1 (pointing down) when upright
    upright_reward = torch.clamp(projected_gravity[:, 2] + 1.0, 0.0, 1.0)
    balance_reward = upright_reward

    # Reward 3: Smoothness reward (penalize jerky motion)
    # Use joint velocity magnitude as proxy for smoothness
    if waist_pitch_idx is not None:
        waist_vel = torch.abs(joint_vel[:, waist_pitch_idx])
        smoothness_reward = torch.exp(-0.5 * waist_vel)
    else:
        smoothness_reward = torch.ones(num_envs, device=device)

    # Combine rewards
    total_reward = (
        angle_reward_weight * angle_reward +
        balance_reward_weight * balance_reward +
        smoothness_reward_weight * smoothness_reward
    )

    # Normalize by total weight
    total_weight = angle_reward_weight + balance_reward_weight + smoothness_reward_weight
    total_reward = total_reward / total_weight

    # Penalize falling (base height too low)
    base_height = robot.data.root_pos_w[:, 2]
    fallen = base_height < 0.3  # If base is below 0.3m, consider fallen
    total_reward = torch.where(fallen, torch.zeros_like(total_reward) - 1.0, total_reward)

    return total_reward


def check_bow_termination(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_base_height: float = 0.3,
) -> torch.Tensor:
    """
    Check if episode should terminate (robot has fallen).

    Returns:
        torch.Tensor: Boolean tensor indicating termination for each env [num_envs]
    """
    robot: Articulation = env.scene[robot_cfg.name]

    # Check if robot has fallen
    base_height = robot.data.root_pos_w[:, 2]
    fallen = base_height < min_base_height

    return fallen
