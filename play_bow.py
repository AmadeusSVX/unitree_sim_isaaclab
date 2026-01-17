#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Play script for G1 bowing motion task - visualize trained policy.

Usage:
    python play_bow.py --checkpoint logs/rsl_rl/g1_bow/*/model_1500.pt
    python play_bow.py --checkpoint logs/rsl_rl/g1_bow/*/model_1500.pt --num_envs 4
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
parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to run")

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

# Import standalone bow task (completely independent of tasks package)
from bow_task import BowG129PPORunnerCfg, BowG129EnvCfg

# Import RSL-RL
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def main():
    """Main play function."""
    print("=" * 60)
    print("G1 Bowing Motion - Play Mode")
    print("=" * 60)

    # Configure environment
    env_cfg = BowG129EnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device

    # Create environment
    env = gym.make("Isaac-Bow-G129-v0", cfg=env_cfg)

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

    # Run simulation
    print(f"\nRunning for {args.num_steps} steps...")
    print("Press Ctrl+C to exit")

    obs, _ = env.get_observations()
    step = 0

    try:
        while simulation_app.is_running() and step < args.num_steps:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

            step += 1
            if step % 100 == 0:
                print(f"Step: {step}/{args.num_steps}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    print("Done!")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
