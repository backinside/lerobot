#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class SOFollowerConfig:
    """Base configuration class for SO Follower robots."""

    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = True


@dataclass
class SOFollowerRobotConfig(RobotConfig, SOFollowerConfig):
    """Base config class for local SO follower robots."""


@RobotConfig.register_subclass("so100_follower")
@dataclass
class SO100FollowerConfig(SOFollowerRobotConfig):
    pass


@RobotConfig.register_subclass("so101_follower")
@dataclass
class SO101FollowerConfig(SOFollowerRobotConfig):
    pass


@dataclass
class SOFollowerClientConfig(RobotConfig):
    """Configuration for a remote SO follower client."""

    remote_ip: str
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5


@RobotConfig.register_subclass("so100_follower_client")
@dataclass
class SO100FollowerClientConfig(SOFollowerClientConfig):
    pass


@RobotConfig.register_subclass("so101_follower_client")
@dataclass
class SO101FollowerClientConfig(SOFollowerClientConfig):
    pass


@dataclass
class SOFollowerHostConfig:
    """Configuration for a remote SO follower host."""

    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    connection_time_s: int = 30
    max_loop_freq_hz: int = 30

SO100FollowerHostConfig = SOFollowerHostConfig
SO101FollowerHostConfig = SOFollowerHostConfig
