# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Abstract base class for robot interface.

Defines the common interface for simulation and real robot control.
"""

from abc import ABC, abstractmethod
import numpy as np


class RobotInterface(ABC):
    """
    Abstract interface for robot control.

    This class defines the common API that both simulation and real robot
    implementations must provide. By using this abstraction, the same
    control logic can work with both Isaac Lab simulation and Unitree SDK.

    Observation space: 43 dimensions
        - Angular velocity (3)
        - Projected gravity (3)
        - Bow angle command (1)
        - Leg joint positions (12)
        - Leg joint velocities (12)
        - Last action (12)

    Action space: 12 dimensions (leg joints)
    """

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """
        Get the current observation from the robot.

        Returns:
            np.ndarray: Observation vector (43 dimensions)
        """
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, action: np.ndarray):
        """
        Apply an action to the robot.

        Args:
            action: Action vector (12 dimensions) - leg joint position offsets
        """
        raise NotImplementedError

    @abstractmethod
    def set_bow_command(self, angle: float):
        """
        Set the target bow angle command.

        Args:
            angle: Target bow angle in radians (0 = upright, positive = forward bow)
        """
        raise NotImplementedError

    @abstractmethod
    def get_bow_command(self) -> float:
        """
        Get the current (smoothed) bow command value.

        Returns:
            float: Current bow command angle
        """
        raise NotImplementedError

    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the robot/simulation is still running.

        Returns:
            bool: True if running, False if stopped or error
        """
        raise NotImplementedError

    @abstractmethod
    def step(self):
        """
        Advance one control step.

        This may include physics stepping in simulation or
        waiting for the next control cycle on real hardware.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """
        Clean up resources and close the interface.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def max_bow_angle(self) -> float:
        """
        Get the maximum bow angle.

        Returns:
            float: Maximum bow angle in radians
        """
        raise NotImplementedError
