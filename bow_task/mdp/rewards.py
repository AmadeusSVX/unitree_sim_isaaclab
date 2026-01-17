# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Reward functions for standing balance and bowing motion task.

Phase 1: Standing balance rewards (12 leg joints only)
Phase 2: Bowing motion rewards (to be added after balance is stable)
"""
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation

# 12 leg joint names (same as policy.onnx)
LEG_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]


def _get_leg_joint_indices(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get cached leg joint indices."""
    cache_key = "_reward_leg_joint_indices"
    if not hasattr(env, cache_key):
        robot: Articulation = env.scene[robot_cfg.name]
        joint_names = robot.data.joint_names
        indices = []
        for leg_joint in LEG_JOINT_NAMES:
            if leg_joint in joint_names:
                indices.append(joint_names.index(leg_joint))
        setattr(env, cache_key, torch.tensor(indices, device=env.device, dtype=torch.long))
    return getattr(env, cache_key)


# =============================================================================
# Phase 1: Standing Balance Rewards
# =============================================================================

def compute_upright_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward for maintaining upright posture.

    Uses projected gravity vector in body frame.
    When upright, projected_gravity_b = [0, 0, -1].

    Returns:
        torch.Tensor: Upright reward [num_envs]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    projected_gravity = robot.data.projected_gravity_b

    # Z component should be close to -1 when upright
    upright_score = torch.clamp(projected_gravity[:, 2] + 1.0, 0.0, 1.0)

    # Also penalize roll/pitch (x, y components should be near 0)
    tilt_penalty = torch.sqrt(projected_gravity[:, 0]**2 + projected_gravity[:, 1]**2)
    tilt_reward = torch.exp(-3.0 * tilt_penalty)

    # Combine both
    reward = 0.5 * upright_score + 0.5 * tilt_reward

    return reward


def compute_base_stability_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_vel_weight: float = 1.0,
    ang_vel_weight: float = 0.5,
) -> torch.Tensor:
    """
    Reward for keeping the base stable (low velocity).

    Penalizes both linear and angular velocity of the base.

    Returns:
        torch.Tensor: Stability reward [num_envs]
    """
    robot: Articulation = env.scene[robot_cfg.name]

    # Linear velocity (in world frame)
    lin_vel = robot.data.root_lin_vel_w
    lin_vel_magnitude = torch.norm(lin_vel, dim=-1)
    lin_vel_reward = torch.exp(-2.0 * lin_vel_magnitude)

    # Angular velocity (in body frame)
    ang_vel = robot.data.root_ang_vel_b
    ang_vel_magnitude = torch.norm(ang_vel, dim=-1)
    ang_vel_reward = torch.exp(-1.0 * ang_vel_magnitude)

    # Weighted combination
    total_weight = lin_vel_weight + ang_vel_weight
    reward = (lin_vel_weight * lin_vel_reward + ang_vel_weight * ang_vel_reward) / total_weight

    return reward


def compute_joint_default_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward for keeping leg joints near their default positions.

    Only evaluates 12 leg joints (not arms or waist).
    Encourages the robot to maintain a stable standing pose.

    Returns:
        torch.Tensor: Joint default position reward [num_envs]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    indices = _get_leg_joint_indices(env, robot_cfg)

    # Get current and default positions for leg joints only
    joint_pos = robot.data.joint_pos[:, indices]
    default_joint_pos = robot.data.default_joint_pos[:, indices]

    # Calculate deviation from default (12 leg joints only)
    joint_deviation = torch.sum(torch.abs(joint_pos - default_joint_pos), dim=-1)

    # Exponential reward (higher when closer to default)
    reward = torch.exp(-0.5 * joint_deviation)

    return reward


# =============================================================================
# Phase 2: Bowing Motion Rewards (for future use)
# =============================================================================

def compute_bow_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_bow_angle: float = 0.5,
    bow_duration_steps: int = 100,
    hold_duration_steps: int = 50,
    return_duration_steps: int = 100,
    angle_reward_weight: float = 2.0,
    balance_reward_weight: float = 0.3,
    smoothness_reward_weight: float = 0.1,
    other_joints_penalty_weight: float = 0.5,
) -> torch.Tensor:
    """
    Compute reward for bowing motion using ONLY waist_pitch_joint.

    The reward encourages:
    - waist_pitch_joint follows the target bow angle trajectory
    - Other joints stay near their default positions
    - Maintaining balance
    - Smooth motion

    Args:
        env: The environment
        robot_cfg: Robot configuration
        target_bow_angle: Target bowing angle in radians (~0.5 = 30 degrees)
        bow_duration_steps: Steps to complete the bow
        hold_duration_steps: Steps to hold the bow position
        return_duration_steps: Steps to return to standing

    Returns:
        torch.Tensor: Reward [num_envs]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device

    # Get joint data
    joint_names = robot.data.joint_names
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel

    # Find waist_pitch_joint index - this is the ONLY joint for bowing
    waist_pitch_idx = None
    for i, name in enumerate(joint_names):
        if "waist_pitch" in name:
            waist_pitch_idx = i
            break

    if waist_pitch_idx is None:
        # Fallback: return zero reward if joint not found
        return torch.zeros(num_envs, device=device)

    # Current waist pitch angle (positive = forward bow)
    current_bow_angle = joint_pos[:, waist_pitch_idx]

    # Calculate target angle based on phase
    total_cycle_steps = bow_duration_steps + hold_duration_steps + return_duration_steps

    if hasattr(env, 'episode_length_buf'):
        current_step = env.episode_length_buf
    else:
        current_step = torch.zeros(num_envs, device=device, dtype=torch.long)

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

    # Phase 2: Hold (bow_duration_steps to bow_duration_steps + hold_duration_steps)
    in_hold_phase = (step_in_cycle >= bow_duration_steps) & (step_in_cycle < bow_duration_steps + hold_duration_steps)
    target_angle = torch.where(
        in_hold_phase,
        torch.full((num_envs,), target_bow_angle, device=device),
        target_angle
    )

    # Phase 3: Return to standing
    in_return_phase = step_in_cycle >= (bow_duration_steps + hold_duration_steps)
    return_progress = (step_in_cycle - bow_duration_steps - hold_duration_steps).float() / return_duration_steps
    return_progress = torch.clamp(return_progress, 0.0, 1.0)
    target_angle = torch.where(
        in_return_phase,
        target_bow_angle * (1.0 - return_progress),
        target_angle
    )

    # === Reward 1: Waist pitch angle tracking (MAIN REWARD) ===
    angle_error = torch.abs(current_bow_angle - target_angle)
    angle_reward = torch.exp(-5.0 * angle_error)  # Sharper reward

    # === Reward 2: Penalize other joints moving (keep them still) ===
    # Find indices of joints that should NOT move much
    leg_joint_mask = torch.zeros(len(joint_names), dtype=torch.bool, device=device)
    for i, name in enumerate(joint_names):
        if any(x in name for x in ["hip", "knee", "ankle"]):
            leg_joint_mask[i] = True

    # Penalize leg joint movement from default
    leg_joint_deviation = torch.sum(torch.abs(joint_pos[:, leg_joint_mask]), dim=-1)
    other_joints_penalty = torch.exp(-0.5 * leg_joint_deviation)

    # === Reward 3: Balance (feet stay on ground) ===
    base_height = robot.data.root_pos_w[:, 2]
    # Reward for maintaining stable height (around 0.8m)
    height_error = torch.abs(base_height - 0.80)
    balance_reward = torch.exp(-5.0 * height_error)

    # === Reward 4: Smoothness (low velocity) ===
    waist_vel = torch.abs(joint_vel[:, waist_pitch_idx])
    smoothness_reward = torch.exp(-0.5 * waist_vel)

    # Combine rewards
    total_reward = (
        angle_reward_weight * angle_reward +
        other_joints_penalty_weight * other_joints_penalty +
        balance_reward_weight * balance_reward +
        smoothness_reward_weight * smoothness_reward
    )

    total_weight = angle_reward_weight + other_joints_penalty_weight + balance_reward_weight + smoothness_reward_weight
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


