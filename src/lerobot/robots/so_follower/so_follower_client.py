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
from functools import cached_property

import cv2
import numpy as np

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_so_follower import SOFollowerClientConfig


class SOFollowerClient(Robot):
    """Remote client for SO-100/SO-101 follower arms hosted over ZMQ."""

    config_class = SOFollowerClientConfig
    name = "so_follower_client"

    def __init__(self, config: SOFollowerClientConfig):
        import zmq

        self._zmq = zmq
        super().__init__(config)
        self.config = config

        self.remote_ip = config.remote_ip
        self.port_zmq_cmd = config.port_zmq_cmd
        self.port_zmq_observations = config.port_zmq_observations
        self.polling_timeout_ms = config.polling_timeout_ms
        self.connect_timeout_s = config.connect_timeout_s

        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None

        self.last_frames: dict[str, np.ndarray] = {}
        self.last_remote_state: RobotObservation = self._make_default_observation()
        self._is_connected = False

    @cached_property
    def _state_ft(self) -> dict[str, type]:
        return {
            "shoulder_pan.pos": float,
            "shoulder_lift.pos": float,
            "elbow_flex.pos": float,
            "wrist_flex.pos": float,
            "wrist_roll.pos": float,
            "gripper.pos": float,
        }

    @cached_property
    def _state_order(self) -> tuple[str, ...]:
        return tuple(self._state_ft.keys())

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def _make_default_observation(self) -> RobotObservation:
        state_vec = np.zeros(len(self._state_order), dtype=np.float32)
        obs_dict: RobotObservation = {key: 0.0 for key in self._state_order}
        obs_dict[OBS_STATE] = state_vec
        return obs_dict

    @check_if_already_connected
    def connect(self) -> None:
        zmq = self._zmq
        self.zmq_context = zmq.Context()

        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_cmd_socket.connect(f"tcp://{self.remote_ip}:{self.port_zmq_cmd}")
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_observation_socket.connect(f"tcp://{self.remote_ip}:{self.port_zmq_observations}")
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)

        self._is_connected = True
        logging.info(
            "Connected remote SO follower client to tcp://%s:%s and tcp://%s:%s",
            self.remote_ip,
            self.port_zmq_cmd,
            self.remote_ip,
            self.port_zmq_observations,
        )

    def calibrate(self) -> None:
        pass

    def _poll_and_get_latest_message(self) -> str | None:
        zmq = self._zmq
        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)

        try:
            socks = dict(poller.poll(self.polling_timeout_ms))
        except zmq.ZMQError as e:
            logging.error(f"ZMQ polling error: {e}")
            return None

        if self.zmq_observation_socket not in socks:
            logging.info("No new data available within timeout.")
            return None

        last_msg = None
        while True:
            try:
                last_msg = self.zmq_observation_socket.recv_string(zmq.NOBLOCK)
            except zmq.Again:
                break

        return last_msg

    def _parse_observation_json(self, obs_string: str) -> RobotObservation | None:
        try:
            return json.loads(obs_string)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON observation: {e}")
            return None

    def _decode_image_from_b64(self, image_b64: str) -> np.ndarray | None:
        if not image_b64:
            return None
        try:
            jpg_data = base64.b64decode(image_b64)
            np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                logging.warning("cv2.imdecode returned None for an image.")
            return frame
        except (TypeError, ValueError) as e:
            logging.error(f"Error decoding base64 image data: {e}")
            return None

    def _remote_state_from_obs(
        self, observation: RobotObservation
    ) -> tuple[dict[str, np.ndarray], RobotObservation]:
        flat_state = {key: observation.get(key, 0.0) for key in self._state_order}
        state_vec = np.array([flat_state[key] for key in self._state_order], dtype=np.float32)
        obs_dict: RobotObservation = {**flat_state, OBS_STATE: state_vec}

        current_frames: dict[str, np.ndarray] = {}
        for cam_name in self._cameras_ft:
            image_b64 = observation.get(cam_name)
            if not isinstance(image_b64, str):
                continue
            frame = self._decode_image_from_b64(image_b64)
            if frame is not None:
                current_frames[cam_name] = frame

        return current_frames, obs_dict

    def _get_data(self) -> tuple[dict[str, np.ndarray], RobotObservation]:
        latest_message_str = self._poll_and_get_latest_message()
        if latest_message_str is None:
            return self.last_frames, self.last_remote_state

        observation = self._parse_observation_json(latest_message_str)
        if observation is None:
            return self.last_frames, self.last_remote_state

        try:
            new_frames, new_state = self._remote_state_from_obs(observation)
        except Exception as e:
            logging.error(f"Error processing observation data, serving last observation: {e}")
            return self.last_frames, self.last_remote_state

        self.last_frames = new_frames
        self.last_remote_state = new_state
        return new_frames, new_state

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        frames, obs_dict = self._get_data()

        for cam_name, frame in frames.items():
            obs_dict[cam_name] = frame

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        self.zmq_cmd_socket.send_string(json.dumps(action))

        actions = np.array([action.get(k, 0.0) for k in self._state_order], dtype=np.float32)
        action_sent = {key: actions[i] for i, key in enumerate(self._state_order)}
        action_sent[ACTION] = actions
        return action_sent

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.zmq_observation_socket is not None:
            self.zmq_observation_socket.close()
        if self.zmq_cmd_socket is not None:
            self.zmq_cmd_socket.close()
        if self.zmq_context is not None:
            self.zmq_context.term()
        self._is_connected = False


SO100FollowerClient = SOFollowerClient
SO101FollowerClient = SOFollowerClient
