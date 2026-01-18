# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Simulation robot interface using Isaac Lab.

This module provides the SimRobotInterface class that wraps Isaac Lab
environment to provide the common RobotInterface API.
"""

import torch
import numpy as np
from typing import Any

from .base import RobotInterface


class SimRobotInterface(RobotInterface):
    """
    Robot interface implementation for Isaac Lab simulation.

    Wraps the Isaac Lab environment and RSL-RL policy to provide
    the common RobotInterface API.
    """

    def __init__(
        self,
        env: Any,
        base_env: Any,
        policy: Any,
        simulation_app: Any,
        device: str = "cpu",
    ):
        """
        Initialize the simulation robot interface.

        Args:
            env: Wrapped environment (RslRlVecEnvWrapper)
            base_env: Unwrapped base environment for command access
            policy: Trained inference policy
            simulation_app: Isaac Sim application instance
            device: Torch device
        """
        self.env = env
        self.base_env = base_env
        self.policy = policy
        self.simulation_app = simulation_app
        self.device = device

        # Get initial observation
        self._obs, _ = self.env.get_observations()

        # Cache bow command term
        self._bow_command_term = self.base_env.command_manager.get_term("bow_angle")

        # Get max bow angle from config
        self._max_bow_angle = self.base_env.cfg.commands.bow_angle.ranges.max_angle

    def get_observation(self) -> np.ndarray:
        """
        Get the current observation.

        Returns:
            np.ndarray: Observation vector (43 dimensions)
        """
        if isinstance(self._obs, torch.Tensor):
            return self._obs[0].cpu().numpy()
        return self._obs[0]

    def apply_action(self, action: np.ndarray):
        """
        Apply an action and step the simulation.

        Args:
            action: Action vector (12 dimensions)
        """
        # Convert to tensor if needed
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)

        # Ensure correct shape [1, 12] for single env
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # Step the environment
        with torch.inference_mode():
            self._obs, _, dones, _ = self.env.step(action)

    def run_policy(self, obs: np.ndarray) -> np.ndarray:
        """
        Run the policy to get action from observation.

        Args:
            obs: Observation vector

        Returns:
            np.ndarray: Action vector
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.inference_mode():
            action = self.policy(obs)

        if isinstance(action, torch.Tensor):
            return action[0].cpu().numpy()
        return action[0]

    def set_bow_command(self, angle: float):
        """
        Set the target bow angle command.

        Args:
            angle: Target bow angle in radians
        """
        self._bow_command_term.target_angle.fill_(angle)

    def get_bow_command(self) -> float:
        """
        Get the current (smoothed) bow command value.

        Returns:
            float: Current bow command angle
        """
        return self._bow_command_term.command[0, 0].item()

    def is_running(self) -> bool:
        """
        Check if the simulation is still running.

        Returns:
            bool: True if simulation is running
        """
        return self.simulation_app.is_running()

    def step(self):
        """
        Advance one control step (already done in apply_action for simulation).
        """
        # In simulation, stepping is handled in apply_action
        pass

    def close(self):
        """
        Clean up and close the simulation.
        """
        self.env.close()
        self.simulation_app.close()

    @property
    def max_bow_angle(self) -> float:
        """
        Get the maximum bow angle.

        Returns:
            float: Maximum bow angle in radians
        """
        return self._max_bow_angle
