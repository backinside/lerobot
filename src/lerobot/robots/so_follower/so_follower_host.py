#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import base64
import json
import logging
import time
from dataclasses import dataclass, field

import cv2
import draccus
import zmq

from .config_so_follower import SOFollowerHostConfig, SOFollowerRobotConfig
from .so_follower import SOFollower


@dataclass
class SOFollowerServerConfig:
    """Configuration for the SO follower host script."""

    robot: SOFollowerRobotConfig = field(default_factory=lambda: SOFollowerRobotConfig(port="/dev/ttyACM0"))
    host: SOFollowerHostConfig = field(default_factory=SOFollowerHostConfig)


class SOFollowerHost:
    def __init__(self, config: SOFollowerHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self) -> None:
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


def _encode_observation(observation: dict[str, object], camera_names: list[str]) -> dict[str, object]:
    encoded = dict(observation)
    for cam_key in camera_names:
        frame = observation.get(cam_key)
        if frame is None:
            encoded[cam_key] = ""
            continue
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        encoded[cam_key] = base64.b64encode(buffer).decode("utf-8") if ret else ""
    return encoded


@draccus.wrap()
def main(cfg: SOFollowerServerConfig):
    logging.info("Configuring SO follower")
    robot = SOFollower(cfg.robot)

    logging.info("Connecting SO follower")
    robot.connect()

    logging.info("Starting SO follower host")
    host = SOFollowerHost(cfg.host)

    logging.info("Waiting for commands...")
    try:
        start = time.perf_counter()
        duration = 0.0
        run_forever = host.connection_time_s == -1
        while run_forever or duration < host.connection_time_s:
            loop_start_time = time.time()

            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                action = dict(json.loads(msg))
                robot.send_action(action)
            except zmq.Again:
                pass
            except Exception as e:
                logging.error("Message fetching failed: %s", e)

            observation = robot.get_observation()
            encoded_observation = _encode_observation(observation, list(robot.cameras))

            try:
                host.zmq_observation_socket.send_string(json.dumps(encoded_observation), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.info("Dropping observation, no client connected")

            elapsed = time.time() - loop_start_time
            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
            duration = time.perf_counter() - start

        if not run_forever:
            print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down SO follower host.")
        robot.disconnect()
        host.disconnect()

    logging.info("Finished SO follower host cleanly")


if __name__ == "__main__":
    main()
