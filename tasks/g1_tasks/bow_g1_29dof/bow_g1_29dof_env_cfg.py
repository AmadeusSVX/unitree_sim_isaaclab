# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Environment configuration for G1 robot bowing motion task.

The robot learns to:
1. Start in standing position
2. Bow forward by bending waist/hip joints
3. Return to standing position
"""
import torch
from dataclasses import MISSING

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg

from . import mdp
from tasks.common_config import G1RobotPresets, CameraPresets
from tasks.common_scene.base_scene_bow import BowSceneCfg


##
# Scene definition
##
@configclass
class BowSceneWithRobotCfg(BowSceneCfg):
    """
    Scene configuration with G1 robot for bowing task.
    Robot is placed standing on the ground plane.
    """

    # G1 robot with wholebody control (includes waist joints)
    # Standing position at origin, facing forward
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex1_wholebody(
        init_pos=(0.0, 0.0, 0.8),  # Standing height
        init_rot=(1.0, 0.0, 0.0, 0.0)  # No rotation
    )

    # Contact sensor for detecting falls
    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
        debug_vis=False
    )

    # Optional: front camera for observation
    front_camera = CameraPresets.g1_front_camera()


##
# MDP settings
##
@configclass
class ActionsCfg:
    """
    Action configuration - control specific joints for bowing.
    Focus on waist and hip pitch joints for bowing motion.
    """
    # Control all joints but the policy will learn to use only relevant ones
    joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],  # All joints
        scale=1.0,
        use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """
    Observation configuration for bowing task.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """
        Policy observation group - information available to the policy.
        """
        # Joint positions and velocities
        joint_pos = ObsTerm(func=mdp.get_joint_positions)
        joint_vel = ObsTerm(func=mdp.get_joint_velocities)

        # Base orientation (for balance)
        base_orientation = ObsTerm(func=mdp.get_base_orientation)

        # Phase information (for timing the bow)
        phase = ObsTerm(func=mdp.get_bow_phase)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True  # Concatenate for MLP policy

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """
    Reward configuration for bowing task.
    """
    # Main bowing reward
    bow_reward = RewTerm(
        func=mdp.compute_bow_reward,
        weight=1.0,
        params={
            "target_bow_angle": 0.5,  # ~30 degrees
            "bow_duration_steps": 100,
            "hold_duration_steps": 50,
            "return_duration_steps": 100,
        }
    )

    # Alive bonus - reward for not falling
    alive_bonus = RewTerm(
        func=mdp.compute_alive_reward,
        weight=0.1
    )

    # Action smoothness penalty
    action_rate = RewTerm(
        func=base_mdp.action_rate_l2,
        weight=-0.01
    )


@configclass
class TerminationsCfg:
    """
    Termination conditions for bowing task.
    """
    # Terminate if robot falls
    fallen = DoneTerm(
        func=mdp.check_fallen,
        params={"min_base_height": 0.3}
    )

    # Time limit
    time_out = DoneTerm(
        func=base_mdp.time_out,
        time_out=True
    )


@configclass
class EventCfg:
    """
    Event configuration - reset events.
    """
    # Reset robot to standing position
    reset_robot = EventTermCfg(
        func=base_mdp.reset_scene_to_default,
        mode="reset",
    )


##
# Main Environment Configuration
##
@configclass
class BowG129EnvCfg(ManagerBasedRLEnvCfg):
    """
    Environment configuration for G1 bowing motion task.
    """

    # Scene configuration
    scene: BowSceneWithRobotCfg = BowSceneWithRobotCfg(
        num_envs=4096,  # Parallel environments for training
        env_spacing=2.5,
        replicate_physics=True
    )

    # MDP configuration
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # No commands needed for this task
    commands = None
    curriculum = None

    def __post_init__(self):
        """Post initialization - simulation settings."""
        # Simulation timing
        self.decimation = 4  # Policy runs at 50 Hz (200 Hz physics / 4)
        self.episode_length_s = 10.0  # 10 second episodes (4 bow cycles)

        # Physics settings
        self.sim.dt = 0.005  # 200 Hz physics
        self.sim.render_interval = self.decimation

        # Contact sensor update
        self.scene.contact_forces.update_period = self.sim.dt

        # PhysX settings for stability
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Physics material properties
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"
