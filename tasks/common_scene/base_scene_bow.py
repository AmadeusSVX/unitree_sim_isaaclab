# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Base scene configuration for bowing motion task.
Simple scene with robot standing on ground plane.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
import os

project_root = os.environ.get("PROJECT_ROOT")


@configclass
class BowSceneCfg(InteractiveSceneCfg):
    """
    Simple scene configuration for bowing motion task.
    Contains only the robot, ground plane, and lighting.
    """

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
        spawn=GroundPlaneCfg(),
    )

    # Dome light for illumination
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0
        ),
    )

    # Distant light for shadows
    distant_light = AssetBaseCfg(
        prim_path="/World/distant_light",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=1000.0,
            angle=0.53,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 10.0],
            rot=[0.7071, 0.0, 0.7071, 0.0],
        ),
    )
