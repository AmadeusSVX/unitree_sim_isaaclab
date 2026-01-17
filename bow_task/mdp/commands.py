# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Command generators for bowing motion task.
"""
import torch
from typing import Sequence
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class BowAngleCommand(CommandTerm):
    """
    Command term for generating target bow angles with smooth transitions.

    The command is a single value representing the target upper body tilt angle
    for bowing. Positive values = forward bow.

    Features:
    - Smooth interpolation between target angles (no sudden jumps)
    - Configurable smoothing rate
    - Resampling at specified intervals
    """

    cfg: "BowAngleCommandCfg"

    def __init__(self, cfg: "BowAngleCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # Current command (smoothed): [num_envs, 1]
        self.bow_angle_command = torch.zeros(self.num_envs, 1, device=self.device)

        # Target command (resampled): [num_envs, 1]
        self.target_angle = torch.zeros(self.num_envs, 1, device=self.device)

    def __str__(self) -> str:
        return (
            f"BowAngleCommand("
            f"range=[{self.cfg.ranges.min_angle}, {self.cfg.ranges.max_angle}], "
            f"smoothing={self.cfg.smoothing_alpha})"
        )

    @property
    def command(self) -> torch.Tensor:
        """Return the current (smoothed) command."""
        return self.bow_angle_command

    def _update_metrics(self):
        """Update metrics (not used for this simple command)."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the TARGET command for specified environments."""
        num_envs = len(env_ids)

        # Randomly sample new target bow angles within the configured range
        min_angle = self.cfg.ranges.min_angle
        max_angle = self.cfg.ranges.max_angle

        # Set the TARGET (not the current command)
        self.target_angle[env_ids, 0] = (
            torch.rand(num_envs, device=self.device) * (max_angle - min_angle) + min_angle
        )

    def _update_command(self):
        """
        Update command with smooth interpolation toward target.

        Called every step. Gradually moves current command toward target.
        """
        alpha = self.cfg.smoothing_alpha

        # Exponential smoothing: command = (1 - alpha) * command + alpha * target
        self.bow_angle_command = (
            (1.0 - alpha) * self.bow_angle_command + alpha * self.target_angle
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization."""
        pass

    def _debug_vis_callback(self, event):
        """Debug visualization callback."""
        pass


@configclass
class BowAngleCommandCfg(CommandTermCfg):
    """Configuration for bow angle command generator."""

    class_type: type = BowAngleCommand

    @configclass
    class Ranges:
        """Range configuration for bow angle command."""
        min_angle: float = 0.0    # Minimum bow angle (standing upright)
        max_angle: float = 0.5    # Maximum bow angle (~30 degrees forward)

    ranges: Ranges = Ranges()

    # Resampling configuration
    resampling_time_range: tuple[float, float] = (3.0, 5.0)  # Resample target every 3-5 seconds

    # Smoothing configuration
    # alpha = 0.02: ~50 steps to reach 63% of target (smooth)
    # alpha = 0.05: ~20 steps to reach 63% of target (moderate)
    # alpha = 0.1:  ~10 steps to reach 63% of target (fast)
    smoothing_alpha: float = 0.02
