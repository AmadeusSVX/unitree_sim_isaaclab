# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Environment configuration for G1 robot bowing motion task.
Standalone version without dependencies on tasks.common_config.
"""
import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg

from . import mdp

# Import robot configuration from robots module
from robots.unitree import G129_CFG_WITH_DEX1_WHOLEBODY


##
# Scene definition
##
@configclass
class BowSceneCfg(InteractiveSceneCfg):
    """Scene configuration for bowing task."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
        spawn=GroundPlaneCfg(),
    )

    # Dome light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0
        ),
    )

    # G1 Robot - use the wholebody configuration
    robot: ArticulationCfg = G129_CFG_WITH_DEX1_WHOLEBODY.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.80),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                ".*_hip_pitch_joint": -0.20,
                ".*_knee_joint": 0.42,
                ".*_ankle_pitch_joint": -0.23,
                ".*_elbow_joint": 0.87,
                "left_shoulder_roll_joint": 0.18,
                "left_shoulder_pitch_joint": 0.35,
                "right_shoulder_roll_joint": -0.18,
                "right_shoulder_pitch_joint": 0.35,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Contact sensor
    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
        debug_vis=False
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action configuration - 12 leg joints only (same as policy.onnx)."""
    joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_hip_pitch_joint",
            ".*_hip_roll_joint",
            ".*_hip_yaw_joint",
            ".*_knee_joint",
            ".*_ankle_pitch_joint",
            ".*_ankle_roll_joint",
        ],
        scale=0.25,
        use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation configuration - policy.onnx compatible format."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Angular velocity (3)
        ang_vel = ObsTerm(func=mdp.get_base_angular_velocity)
        # Projected gravity (3)
        projected_gravity = ObsTerm(func=mdp.get_base_orientation)
        # Leg joint positions relative to default (12)
        leg_joint_pos = ObsTerm(func=mdp.get_leg_joint_positions)
        # Leg joint velocities (12)
        leg_joint_vel = ObsTerm(func=mdp.get_leg_joint_velocities)
        # Last action (12)
        last_action = ObsTerm(func=mdp.get_last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward configuration for Phase 1: Standing Balance."""
    # Main balance rewards
    upright = RewTerm(
        func=mdp.compute_upright_reward,
        weight=1.0
    )
    base_stability = RewTerm(
        func=mdp.compute_base_stability_reward,
        weight=0.5
    )
    joint_default = RewTerm(
        func=mdp.compute_joint_default_reward,
        weight=0.3
    )
    # Survival reward
    alive_bonus = RewTerm(
        func=mdp.compute_alive_reward,
        weight=0.2
    )
    # Action smoothness penalty
    action_rate = RewTerm(
        func=base_mdp.action_rate_l2,
        weight=-0.01
    )


@configclass
class TerminationsCfg:
    """Termination configuration."""
    fallen = DoneTerm(
        func=mdp.check_fallen,
        params={"min_base_height": 0.3}
    )
    time_out = DoneTerm(
        func=base_mdp.time_out,
        time_out=True
    )


@configclass
class EventCfg:
    """Event configuration."""
    reset_robot = EventTermCfg(
        func=base_mdp.reset_scene_to_default,
        mode="reset",
    )


##
# Main Environment Configuration
##
@configclass
class BowG129EnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration for G1 bowing motion task."""

    scene: BowSceneCfg = BowSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    commands = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # Contact sensor update period
        self.scene.contact_forces.update_period = self.sim.dt

        # PhysX settings
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Physics material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"
