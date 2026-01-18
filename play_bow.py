#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Play script for G1 bowing motion task - visualize trained policy.

Uses the RobotInterface abstraction layer for sim-to-real compatibility.

Usage:
    python play_bow.py --checkpoint logs/rsl_rl/g1_bow/*/model_1500.pt
    python play_bow.py --checkpoint logs/rsl_rl/g1_bow/*/model_1500.pt --num_envs 4

Controls:
    B key: Hold to bow, release to return to standing
    Ctrl+C: Exit
"""

import os
import argparse
import glob

# Set project root
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root

# Parse arguments before importing Isaac Lab
parser = argparse.ArgumentParser(description="Play G1 Bowing Motion")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (supports wildcards)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--device", type=str, default="cpu", help="Device")
parser.add_argument("--num_steps", type=int, default=0, help="Number of steps to run (0 for infinite)")

args = parser.parse_args()

# Resolve checkpoint path with wildcards
checkpoint_paths = glob.glob(args.checkpoint)
if not checkpoint_paths:
    print(f"Error: No checkpoint found matching: {args.checkpoint}")
    exit(1)
checkpoint_path = sorted(checkpoint_paths)[-1]  # Take the latest
print(f"Using checkpoint: {checkpoint_path}")

# Import Isaac Lab app launcher
from isaaclab.app import AppLauncher

# Configure app launcher (with GUI for visualization)
app_launcher_args = argparse.Namespace(
    headless=False,
    device=args.device,
    enable_cameras=False,
)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

# Now import other modules
import torch
import gymnasium as gym

# Import carb for keyboard input
import carb.input
import omni.appwindow

# Import standalone bow task (completely independent of tasks package)
from bow_task import BowG129PPORunnerCfg, BowG129EnvCfg

# Import RSL-RL
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# Import robot interface abstraction
from robot_interface import RobotInterface, SimRobotInterface


class KeyboardController:
    """Keyboard controller for bow command."""

    def __init__(self):
        self._input = carb.input.acquire_input_interface()
        self._app_window = omni.appwindow.get_default_app_window()
        self._keyboard = self._app_window.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )
        self.bow_pressed = False

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        """Handle keyboard events."""
        # B key for bowing
        if event.input == carb.input.KeyboardInput.B:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                self.bow_pressed = True
            elif event.type == carb.input.KeyboardEventType.KEY_REPEAT:
                self.bow_pressed = True  # Maintain pressed state on repeat
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                self.bow_pressed = False
        return True

    def close(self):
        """Unsubscribe from keyboard events."""
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)


def run_control_loop(
    robot: RobotInterface,
    keyboard: KeyboardController,
    num_steps: int = 0,
):
    """
    Run the main control loop.

    This is the common control logic that works with both
    simulation and real robot via the RobotInterface abstraction.

    Args:
        robot: Robot interface (SimRobotInterface or RealRobotInterface)
        keyboard: Keyboard controller for user input
        num_steps: Number of steps to run (0 for infinite)
    """
    infinite_mode = num_steps == 0

    if infinite_mode:
        print("\nRunning indefinitely (Ctrl+C to exit)...")
    else:
        print(f"\nRunning for {num_steps} steps...")

    max_bow_angle = robot.max_bow_angle

    print("\n" + "=" * 60)
    print("Controls:")
    print("  [B] Hold to bow, release to stand")
    print("  [Ctrl+C] Exit")
    print(f"\nBow angle: {max_bow_angle:.2f} rad (~{max_bow_angle * 180 / 3.14159:.1f} deg)")
    print("=" * 60 + "\n")

    step = 0
    last_state = None
    bow_hold_counter = 0  # Hold BOW state for minimum steps
    BOW_HOLD_STEPS = 5    # Minimum steps to hold BOW state

    try:
        while robot.is_running() and (infinite_mode or step < num_steps):
            # Debounce: if B is pressed, reset hold counter
            if keyboard.bow_pressed:
                bow_hold_counter = BOW_HOLD_STEPS

            # BOW state is active while hold counter > 0
            is_bowing = bow_hold_counter > 0
            if bow_hold_counter > 0:
                bow_hold_counter -= 1

            target_angle = max_bow_angle if is_bowing else 0.0
            current_state = "BOW" if is_bowing else "STAND"

            # Set the target angle on the robot
            robot.set_bow_command(target_angle)

            # Get observation and run policy
            obs = robot.get_observation()
            action = robot.run_policy(obs)

            # Apply action
            robot.apply_action(action)

            # Step (for timing on real robot)
            robot.step()

            step += 1

            # Print state changes
            if current_state != last_state:
                current_cmd = robot.get_bow_command()
                print(f"Step: {step} | State: {current_state} | Target: {target_angle:.2f} | Current: {current_cmd:.3f}")
                last_state = current_state

            # Periodic status update
            if step % 500 == 0:
                current_cmd = robot.get_bow_command()
                print(f"Step: {step} | State: {current_state} | Target: {target_angle:.2f} | Current: {current_cmd:.3f}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    print("Done!")


def main():
    """Main play function."""
    print("=" * 60)
    print("G1 Bowing Motion - Play Mode")
    print("=" * 60)

    # Configure environment
    env_cfg = BowG129EnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device

    # Disable time_out for playback (only reset on fall)
    env_cfg.terminations.time_out = None

    # Create environment
    env = gym.make("Isaac-Bow-G129-v0", cfg=env_cfg)

    # Get the base environment for command control
    base_env = env.unwrapped

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Configure agent
    agent_cfg = BowG129PPORunnerCfg()

    # Create runner and load checkpoint
    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=args.device,
    )
    runner.load(checkpoint_path)

    # Get inference policy
    policy = runner.get_inference_policy(device=args.device)

    # Create robot interface (abstraction layer)
    robot = SimRobotInterface(
        env=env,
        base_env=base_env,
        policy=policy,
        simulation_app=simulation_app,
        device=args.device,
    )

    # Initialize keyboard controller
    keyboard = KeyboardController()

    try:
        # Run control loop (common logic for sim and real)
        run_control_loop(robot, keyboard, num_steps=args.num_steps)
    finally:
        # Cleanup
        keyboard.close()
        robot.close()


if __name__ == "__main__":
    main()
