# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Real robot interface using Unitree SDK.

This module provides the RealRobotInterface class that communicates
with the actual G1 robot via Unitree SDK2.

NOTE: This is a skeleton implementation. Fill in the Unitree SDK
specific code when deploying to real hardware.
"""

import time
import numpy as np
from typing import Optional

from .base import RobotInterface


# Leg joint indices mapping from simulation to real robot motor IDs
# TODO: Verify these mappings with actual robot configuration
LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

# TODO: Map to actual Unitree motor IDs
REAL_MOTOR_IDS = [
    8, 0,    # hip_pitch (left, right)
    9, 1,    # hip_roll
    10, 2,   # hip_yaw
    11, 3,   # knee
    12, 4,   # ankle_pitch
    13, 5,   # ankle_roll
]

# Default joint positions (same as simulation)
DEFAULT_JOINT_POS = np.array([
    -0.20, -0.20,  # hip_pitch
    0.0, 0.0,      # hip_roll
    0.0, 0.0,      # hip_yaw
    0.42, 0.42,    # knee
    -0.23, -0.23,  # ankle_pitch
    0.0, 0.0,      # ankle_roll
])

# Control parameters
ACTION_SCALE = 0.25
CONTROL_FREQUENCY = 50  # Hz (matches simulation: dt=0.005 * decimation=4)
POSITION_KP = 50.0
POSITION_KD = 1.0


class RealRobotInterface(RobotInterface):
    """
    Robot interface implementation for real Unitree G1 robot.

    Uses Unitree SDK2 to communicate with the robot hardware.

    NOTE: This is a skeleton implementation. The actual Unitree SDK
    calls need to be implemented based on the SDK documentation.
    """

    def __init__(
        self,
        policy_path: str,
        obs_mean_path: Optional[str] = None,
        obs_std_path: Optional[str] = None,
        max_bow_angle: float = 0.25,
    ):
        """
        Initialize the real robot interface.

        Args:
            policy_path: Path to the ONNX policy file
            obs_mean_path: Path to observation mean for normalization
            obs_std_path: Path to observation std for normalization
            max_bow_angle: Maximum bow angle in radians
        """
        self._max_bow_angle = max_bow_angle
        self._running = False
        self._emergency_stop = False

        # Load ONNX policy
        self._load_policy(policy_path)

        # Load normalization parameters
        self._load_normalization(obs_mean_path, obs_std_path)

        # State tracking
        self._last_action = np.zeros(12)
        self._bow_command = 0.0
        self._smoothed_bow_command = 0.0
        self._smoothing_alpha = 0.02

        # Control timing
        self._control_period = 1.0 / CONTROL_FREQUENCY
        self._last_control_time = time.time()

        # Initialize SDK connection
        self._init_sdk()

    def _load_policy(self, policy_path: str):
        """Load the ONNX policy for inference."""
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(policy_path)
            self._input_name = self._session.get_inputs()[0].name
            print(f"Loaded policy from: {policy_path}")
        except ImportError:
            raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime")
        except Exception as e:
            raise RuntimeError(f"Failed to load policy: {e}")

    def _load_normalization(
        self,
        obs_mean_path: Optional[str],
        obs_std_path: Optional[str]
    ):
        """Load observation normalization parameters."""
        if obs_mean_path and obs_std_path:
            self._obs_mean = np.load(obs_mean_path)
            self._obs_std = np.load(obs_std_path)
            self._normalize = True
            print("Loaded normalization parameters")
        else:
            self._obs_mean = np.zeros(43)
            self._obs_std = np.ones(43)
            self._normalize = False
            print("WARNING: No normalization parameters provided")

    def _init_sdk(self):
        """
        Initialize Unitree SDK connection.

        TODO: Implement actual SDK initialization:
        - Initialize DDS communication
        - Subscribe to LowState
        - Create LowCmd publisher
        """
        print("WARNING: Unitree SDK initialization not implemented")
        print("TODO: Add actual SDK code for real robot deployment")

        # Placeholder for SDK objects
        # self._state_subscriber = ...
        # self._cmd_publisher = ...
        # self._latest_state = None

        self._running = True

    def _get_imu_data(self) -> dict:
        """
        Get IMU data from the robot.

        TODO: Implement actual SDK call to get IMU data.

        Returns:
            dict with 'quaternion' and 'gyroscope'
        """
        # Placeholder - return default values
        return {
            "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),  # [w, x, y, z]
            "gyroscope": np.array([0.0, 0.0, 0.0]),  # [x, y, z] rad/s
        }

    def _get_joint_states(self) -> tuple:
        """
        Get joint positions and velocities from the robot.

        TODO: Implement actual SDK call to get motor states.

        Returns:
            tuple: (positions, velocities) for leg joints
        """
        # Placeholder - return default values
        positions = DEFAULT_JOINT_POS.copy()
        velocities = np.zeros(12)
        return positions, velocities

    def _compute_projected_gravity(self, quaternion: np.ndarray) -> np.ndarray:
        """
        Compute projected gravity from quaternion.

        Args:
            quaternion: [w, x, y, z] orientation

        Returns:
            np.ndarray: Projected gravity vector [x, y, z]
        """
        w, x, y, z = quaternion

        # Rotation matrix from quaternion
        # Gravity in world frame is [0, 0, -1]
        # Project to body frame
        gx = 2.0 * (x * z - w * y)
        gy = 2.0 * (y * z + w * x)
        gz = 1.0 - 2.0 * (x * x + y * y)

        return np.array([-gx, -gy, -gz])

    def get_observation(self) -> np.ndarray:
        """
        Get the current observation from real robot sensors.

        Returns:
            np.ndarray: Observation vector (43 dimensions)
        """
        obs = np.zeros(43)

        # Get sensor data
        imu = self._get_imu_data()
        positions, velocities = self._get_joint_states()

        # Angular velocity (scaled by 0.25)
        obs[0:3] = imu["gyroscope"] * 0.25

        # Projected gravity
        obs[3:6] = self._compute_projected_gravity(imu["quaternion"])

        # Bow angle command (smoothed)
        obs[6] = self._smoothed_bow_command

        # Leg joint positions (relative to default)
        obs[7:19] = positions - DEFAULT_JOINT_POS

        # Leg joint velocities (scaled by 0.05)
        obs[19:31] = velocities * 0.05

        # Last action
        obs[31:43] = self._last_action

        # Normalize if parameters are available
        if self._normalize:
            obs = (obs - self._obs_mean) / (self._obs_std + 1e-8)

        return obs.astype(np.float32)

    def run_policy(self, obs: np.ndarray) -> np.ndarray:
        """
        Run the ONNX policy to get action.

        Args:
            obs: Observation vector

        Returns:
            np.ndarray: Action vector
        """
        inputs = {self._input_name: obs[np.newaxis, :]}
        outputs = self._session.run(None, inputs)
        return outputs[0][0]

    def apply_action(self, action: np.ndarray):
        """
        Apply action to the real robot.

        Args:
            action: Action vector (12 dimensions)
        """
        # Compute target joint positions
        target_positions = DEFAULT_JOINT_POS + action * ACTION_SCALE

        # Send command to robot
        self._send_joint_command(target_positions)

        # Store for next observation
        self._last_action = action.copy()

    def _send_joint_command(self, target_positions: np.ndarray):
        """
        Send joint position command to the robot.

        TODO: Implement actual SDK call to send motor commands.

        Args:
            target_positions: Target joint positions for 12 leg joints
        """
        # Placeholder - print command
        # In real implementation:
        # cmd = LowCmd_()
        # for i, pos in enumerate(target_positions):
        #     motor_id = REAL_MOTOR_IDS[i]
        #     cmd.motor_cmd[motor_id].mode = 0x01
        #     cmd.motor_cmd[motor_id].q = pos
        #     cmd.motor_cmd[motor_id].kp = POSITION_KP
        #     cmd.motor_cmd[motor_id].kd = POSITION_KD
        # self._cmd_publisher.Write(cmd)
        pass

    def set_bow_command(self, angle: float):
        """
        Set the target bow angle command.

        Args:
            angle: Target bow angle in radians
        """
        self._bow_command = angle

    def get_bow_command(self) -> float:
        """
        Get the current (smoothed) bow command value.

        Returns:
            float: Current smoothed bow command angle
        """
        return self._smoothed_bow_command

    def is_running(self) -> bool:
        """
        Check if the robot is still running.

        Returns:
            bool: True if running and no emergency stop
        """
        return self._running and not self._emergency_stop

    def step(self):
        """
        Advance one control step with timing.
        """
        # Update smoothed bow command
        alpha = self._smoothing_alpha
        self._smoothed_bow_command = (
            (1.0 - alpha) * self._smoothed_bow_command +
            alpha * self._bow_command
        )

        # Wait for next control cycle
        current_time = time.time()
        elapsed = current_time - self._last_control_time
        sleep_time = self._control_period - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)

        self._last_control_time = time.time()

    def close(self):
        """
        Clean up and close the SDK connection.
        """
        self._running = False
        print("Robot interface closed")

    def emergency_stop(self):
        """
        Trigger emergency stop.
        """
        self._emergency_stop = True
        print("EMERGENCY STOP ACTIVATED")

    @property
    def max_bow_angle(self) -> float:
        """
        Get the maximum bow angle.

        Returns:
            float: Maximum bow angle in radians
        """
        return self._max_bow_angle
