# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Robot interface abstraction layer for sim-to-real transfer.

This module provides a common interface for controlling the robot
in both simulation (Isaac Lab) and real hardware (Unitree SDK).
"""

from .base import RobotInterface
from .sim_interface import SimRobotInterface
from .real_interface import RealRobotInterface

__all__ = [
    "RobotInterface",
    "SimRobotInterface",
    "RealRobotInterface",
]
