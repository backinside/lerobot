#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import contextlib
import logging
import sys
import threading
import time
from queue import Queue
from typing import Any

if sys.platform == "win32":
    import msvcrt
else:
    import select
    import termios
    import tty

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_keyboard import (
    KeyboardEndEffectorTeleopConfig,
    KeyboardRoverTeleopConfig,
    KeyboardTeleopConfig,
)

KEY_UP = "up"
KEY_DOWN = "down"
KEY_LEFT = "left"
KEY_RIGHT = "right"
KEY_SHIFT = "shift"
KEY_SHIFT_R = "shift_r"
KEY_CTRL_L = "ctrl_l"
KEY_CTRL_R = "ctrl_r"
KEY_ESC = "esc"

ESCAPE_SEQUENCES = {
    "\x1b[A": KEY_UP,
    "\x1b[B": KEY_DOWN,
    "\x1b[D": KEY_LEFT,
    "\x1b[C": KEY_RIGHT,
    "\xe0H": KEY_UP,
    "\xe0P": KEY_DOWN,
    "\xe0K": KEY_LEFT,
    "\xe0M": KEY_RIGHT,
}


class KeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = set()
        self.listener = None
        self.logs = {}
        self._reader_thread = None
        self._stop_event = threading.Event()
        self._stdin_fd = None
        self._stdin_attrs = None
        self._connected = False

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected and self._reader_thread is not None and self._reader_thread.is_alive()

    @property
    def is_calibrated(self) -> bool:
        pass

    @check_if_already_connected
    def connect(self) -> None:
        self._configure_stdin()
        self._stop_event.clear()
        self._reader_thread = threading.Thread(target=self._read_stdin_loop, daemon=True)
        self._reader_thread.start()
        self.listener = self._reader_thread
        self._connected = True
        logging.info("stdin keyboard teleop enabled. Terminal set to raw mode.")

    def calibrate(self) -> None:
        pass

    def _configure_stdin(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("Keyboard teleop requires a TTY stdin.")

        if sys.platform == "win32":
            return

        self._stdin_fd = sys.stdin.fileno()
        self._stdin_attrs = termios.tcgetattr(self._stdin_fd)
        tty.setraw(self._stdin_fd)

    def _restore_stdin(self) -> None:
        if sys.platform == "win32":
            return

        if self._stdin_fd is not None and self._stdin_attrs is not None:
            with contextlib.suppress(termios.error):
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_attrs)
        self._stdin_fd = None
        self._stdin_attrs = None

    def _read_char(self) -> str | None:
        if sys.platform == "win32":
            if not msvcrt.kbhit():
                return None
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                return ch + msvcrt.getwch()
            return ch

        if self._stdin_fd is None:
            return None
        ready, _, _ = select.select([self._stdin_fd], [], [], 0.05)
        if not ready:
            return None
        return sys.stdin.read(1)

    def _normalize_key(self, raw_key: str) -> str | None:
        if raw_key in ESCAPE_SEQUENCES:
            return ESCAPE_SEQUENCES[raw_key]
        if raw_key in ("\x1b", "\x00\x1b", "\xe0\x1b"):
            return KEY_ESC
        if raw_key in ("\x10",):
            return KEY_CTRL_R
        if raw_key in ("\x0c",):
            return KEY_CTRL_L
        if raw_key in ("\x1a",):
            return KEY_SHIFT_R
        if raw_key in ("\x13",):
            return KEY_SHIFT
        if len(raw_key) == 1:
            return raw_key.lower()
        return None

    def _read_stdin_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                raw_key = self._read_char()
                if raw_key is None:
                    continue

                if raw_key == "\x1b" and sys.platform != "win32":
                    next_chars = []
                    for _ in range(2):
                        ready, _, _ = select.select([self._stdin_fd], [], [], 0.001)
                        if not ready:
                            break
                        next_chars.append(sys.stdin.read(1))
                    raw_key = raw_key + "".join(next_chars)

                key = self._normalize_key(raw_key)
                if key is None:
                    continue

                self.event_queue.put(key)
                if key == KEY_ESC:
                    logging.info("ESC pressed, disconnecting.")
                    self._stop_event.set()
                    break
        finally:
            self._connected = False
            self._restore_stdin()

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char = self.event_queue.get_nowait()
            self.current_pressed.add(key_char)

    def configure(self):
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        action = set(self.current_pressed)
        self.current_pressed.clear()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return dict.fromkeys(action, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        self._stop_event.set()
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=0.2)
        self._restore_stdin()
        self._connected = False
        self.listener = None
        self._reader_thread = None
        self.current_pressed.clear()


class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self._drain_pressed_keys()
        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        gripper_action = 1.0

        for key in self.current_pressed:
            if key == KEY_UP:
                delta_y = -1.0
            elif key == KEY_DOWN:
                delta_y = 1.0
            elif key == KEY_LEFT:
                delta_x = 1.0
            elif key == KEY_RIGHT:
                delta_x = -1.0
            elif key == KEY_SHIFT:
                delta_z = -1.0
            elif key == KEY_SHIFT_R:
                delta_z = 1.0
            elif key == KEY_CTRL_R:
                gripper_action = 2.0
            elif key == KEY_CTRL_L:
                gripper_action = 0.0
            else:
                self.misc_keys_queue.put(key)

        self.current_pressed.clear()

        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
        }

        if self.config.use_gripper:
            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the keyboard such as intervention status,
        episode termination, success indicators, etc.

        Keyboard mappings:
        - Any movement keys pressed = intervention active
        - 's' key = success (terminate episode successfully)
        - 'r' key = rerecord episode (terminate and rerecord)
        - 'q' key = quit episode (terminate without success)

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Check if any movement keys are currently pressed (indicates intervention)
        movement_keys = {KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_SHIFT, KEY_SHIFT_R, KEY_CTRL_R, KEY_CTRL_L}
        is_intervention = any(key in movement_keys for key in self.current_pressed)

        # Check for episode control commands from misc_keys_queue
        terminate_episode = False
        success = False
        rerecord_episode = False

        # Process any pending misc keys
        while not self.misc_keys_queue.empty():
            key = self.misc_keys_queue.get_nowait()
            if key == "s":
                success = True
            elif key == "r":
                terminate_episode = True
                rerecord_episode = True
            elif key == "q":
                terminate_episode = True
                success = False

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }


class KeyboardRoverTeleop(KeyboardTeleop):
    """
    Keyboard teleoperator for mobile robots like EarthRover Mini Plus.

    Provides intuitive WASD-style controls for driving a mobile robot:
    - Linear movement (forward/backward)
    - Angular movement (turning/rotation)
    - Speed adjustment
    - Emergency stop

    Keyboard Controls:
        Movement:
            - W: Move forward
            - S: Move backward
            - A: Turn left (with forward motion)
            - D: Turn right (with forward motion)
            - Q: Rotate left in place
            - E: Rotate right in place
            - X: Emergency stop

        Speed Control:
            - +/=: Increase speed
            - -: Decrease speed

        System:
            - ESC: Disconnect teleoperator

    Attributes:
        config: Teleoperator configuration
        current_linear_speed: Current linear velocity magnitude
        current_angular_speed: Current angular velocity magnitude

    Example:
        ```python
        from lerobot.teleoperators.keyboard import KeyboardRoverTeleop, KeyboardRoverTeleopConfig

        teleop = KeyboardRoverTeleop(
            KeyboardRoverTeleopConfig(linear_speed=1.0, angular_speed=1.0, speed_increment=0.1)
        )
        teleop.connect()

        while teleop.is_connected:
            action = teleop.get_action()
            robot.send_action(action)
        ```
    """

    config_class = KeyboardRoverTeleopConfig
    name = "keyboard_rover"

    def __init__(self, config: KeyboardRoverTeleopConfig):
        super().__init__(config)
        # Add rover-specific speed settings
        self.current_linear_speed = config.linear_speed
        self.current_angular_speed = config.angular_speed

    @property
    def action_features(self) -> dict:
        """Return action format for rover (linear and angular velocities)."""
        return {
            "linear_velocity": float,
            "angular_velocity": float,
        }

    @property
    def is_calibrated(self) -> bool:
        """Rover teleop doesn't require calibration."""
        return True

    def _drain_pressed_keys(self):
        """Update current_pressed from queued stdin keypresses."""
        while not self.event_queue.empty():
            key_char = self.event_queue.get_nowait()
            self.current_pressed.add(key_char)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """
        Get the current action based on pressed keys.

        Returns:
            RobotAction with 'linear_velocity' and 'angular_velocity' keys.
        """
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        linear_velocity = 0.0
        angular_velocity = 0.0

        active_keys = set(self.current_pressed)

        # Linear movement (W/S) - these take priority
        if "w" in active_keys:
            linear_velocity = self.current_linear_speed
        elif "s" in active_keys:
            linear_velocity = -self.current_linear_speed

        # Turning (A/D/Q/E)
        if "d" in active_keys:
            angular_velocity = -self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "a" in active_keys:
            angular_velocity = self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "q" in active_keys:
            angular_velocity = self.current_angular_speed
            linear_velocity = 0  # Rotate in place
        elif "e" in active_keys:
            angular_velocity = -self.current_angular_speed
            linear_velocity = 0  # Rotate in place

        # Stop (X) - overrides everything
        if "x" in active_keys:
            linear_velocity = 0
            angular_velocity = 0

        # Speed adjustment
        if "+" in active_keys or "=" in active_keys:
            self.current_linear_speed += self.config.speed_increment
            self.current_angular_speed += self.config.speed_increment * self.config.angular_speed_ratio
            logging.info(
                f"Speed increased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )
        if "-" in active_keys:
            self.current_linear_speed = max(
                self.config.min_linear_speed, self.current_linear_speed - self.config.speed_increment
            )
            self.current_angular_speed = max(
                self.config.min_angular_speed,
                self.current_angular_speed - self.config.speed_increment * self.config.angular_speed_ratio,
            )
            logging.info(
                f"Speed decreased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )

        self.current_pressed.clear()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return {
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
        }
