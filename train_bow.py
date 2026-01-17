#!/usr/bin/env python3
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Training script for G1 bowing motion task using RSL-RL PPO.

Usage:
    python train_bow.py --num_envs 1024 --device cuda:0 --headless
    python train_bow.py --num_envs 64 --device cpu  # For testing
"""

import os
import argparse

# Set project root
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root

# Parse arguments before importing Isaac Lab
parser = argparse.ArgumentParser(description="Train G1 Bowing Motion")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel environments")
parser.add_argument("--device", type=str, default="cuda:0", help="Device (cuda:0 or cpu)")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--max_iterations", type=int, default=1500, help="Maximum training iterations")
parser.add_argument("--save_interval", type=int, default=100, help="Checkpoint save interval")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--log_dir", type=str, default="logs/rsl_rl/g1_bow", help="Log directory")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

args = parser.parse_args()

# Import Isaac Lab app launcher
from isaaclab.app import AppLauncher

# Configure app launcher
app_launcher_args = argparse.Namespace(
    headless=args.headless,
    device=args.device,
    enable_cameras=False,
)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

# Now import other modules
import torch
import gymnasium as gym
from datetime import datetime

# Import standalone bow task (completely independent of tasks package)
from bow_task import BowG129PPORunnerCfg, BowG129EnvCfg

# Import RSL-RL
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def main():
    """Main training function."""
    print("=" * 60)
    print("G1 Bowing Motion Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Num Envs: {args.num_envs}")
    print(f"Headless: {args.headless}")
    print(f"Max Iterations: {args.max_iterations}")
    print("=" * 60)

    # Configure environment
    env_cfg = BowG129EnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed

    # Create environment
    env = gym.make("Isaac-Bow-G129-v0", cfg=env_cfg)

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Configure agent
    agent_cfg = BowG129PPORunnerCfg()
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.save_interval = args.save_interval

    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(project_root, args.log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log directory: {log_dir}")

    # Create runner
    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=args.device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        runner.load(args.resume)

    # Train
    print("\nStarting training...")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    # Save final model
    print("\nTraining complete!")
    print(f"Checkpoints saved to: {log_dir}")

    # Export policy
    export_dir = os.path.join(log_dir, "exported")
    os.makedirs(export_dir, exist_ok=True)

    # Get policy for export
    policy = runner.get_inference_policy(device=args.device)

    # Save as JIT
    try:
        from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

        export_policy_as_jit(
            runner.alg.actor_critic,
            runner.obs_normalizer,
            path=export_dir,
            filename="policy.pt"
        )
        print(f"Exported JIT policy to: {export_dir}/policy.pt")

        export_policy_as_onnx(
            runner.alg.actor_critic,
            normalizer=runner.obs_normalizer,
            path=export_dir,
            filename="policy.onnx"
        )
        print(f"Exported ONNX policy to: {export_dir}/policy.onnx")
    except Exception as e:
        print(f"Warning: Could not export policy: {e}")

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
