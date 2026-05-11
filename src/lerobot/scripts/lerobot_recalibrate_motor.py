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

"""
Interactively recalibrate a single motor on an existing robot or teleoperator calibration.

Requires: pip install 'lerobot[hardware]'
"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    hope_jr,
    koch_follower,
    lekiwi,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    so_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    so_leader,
    unitree_g1,
)
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS, TELEOPERATORS
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass
class RecalibrateMotorConfig:
    teleop: TeleoperatorConfig | None = None
    robot: RobotConfig | None = None

    def __post_init__(self):
        if bool(self.teleop) == bool(self.robot):
            raise ValueError("Choose either a teleop or a robot.")

        self.device = self.robot if self.robot else self.teleop


def _make_device(device_config: RobotConfig | TeleoperatorConfig) -> Robot | Teleoperator:
    if isinstance(device_config, RobotConfig):
        return make_robot_from_config(device_config)
    if isinstance(device_config, TeleoperatorConfig):
        return make_teleoperator_from_config(device_config)
    raise TypeError(f"Unsupported device config type: {type(device_config)}")


def _get_calibration_kind(device: Robot | Teleoperator) -> str:
    return ROBOTS if isinstance(device, Robot) else TELEOPERATORS


def discover_calibration_files(device: Robot | Teleoperator) -> list[Path]:
    calibration_paths: dict[Path, None] = {}

    for path in sorted(device.calibration_dir.glob("*.json")):
        calibration_paths[path] = None

    kind = _get_calibration_kind(device)
    for legacy_name in device.legacy_calibration_names:
        legacy_dir = HF_LEROBOT_CALIBRATION / kind / legacy_name
        for path in sorted(legacy_dir.glob("*.json")):
            calibration_paths[path] = None

    return sorted(calibration_paths)


def _prompt_for_index(prompt: str, options: list[str]) -> int:
    while True:
        user_input = input(prompt).strip()
        if not user_input:
            raise ValueError("Selection is required.")
        if not user_input.isdigit():
            print("Enter the number corresponding to your choice.")
            continue

        idx = int(user_input) - 1
        if idx < 0 or idx >= len(options):
            print("Selection out of range.")
            continue

        return idx


def _choose_calibration_file(device: Robot | Teleoperator) -> Path:
    calibration_files = discover_calibration_files(device)
    if not calibration_files:
        raise FileNotFoundError(
            f"No calibration files found for '{device.name}' in '{device.calibration_dir}'."
        )

    if len(calibration_files) == 1:
        chosen = calibration_files[0]
        print(f"Using calibration: {chosen}")
        return chosen

    print("Available calibrations:")
    labels = []
    for idx, path in enumerate(calibration_files, start=1):
        label = f"[{idx}] {path.stem} ({path})"
        labels.append(label)
        print(label)

    selected = _prompt_for_index("\nSelect a calibration: ", labels)
    return calibration_files[selected]


def _load_selected_calibration(device: Robot | Teleoperator, calibration_fpath: Path) -> None:
    device.id = calibration_fpath.stem
    device.calibration_fpath = calibration_fpath
    device._load_calibration(calibration_fpath)

    bus = getattr(device, "bus", None)
    if bus is not None:
        bus.calibration = device.calibration


def _require_single_bus(device: Robot | Teleoperator) -> Any:
    bus = getattr(device, "bus", None)
    if bus is None:
        raise NotImplementedError(
            f"{device.__class__.__name__} is not supported by this script because it does not expose a single 'bus'."
        )

    if not hasattr(bus, "set_half_turn_homings") or not hasattr(bus, "record_ranges_of_motion"):
        raise NotImplementedError(
            f"{device.__class__.__name__} is not supported by this script because its bus does not use "
            "range-based single-motor calibration."
        )

    return bus


def _is_full_turn_motor(bus: Any, motor: str, calibration: Any) -> bool:
    model_resolution_table = getattr(bus, "model_resolution_table", None)
    if not model_resolution_table:
        return False

    model = bus.motors[motor].model
    max_resolution = model_resolution_table[model] - 1
    return calibration.range_min == 0 and calibration.range_max == max_resolution


def recalibrate_selected_motor(device: Robot | Teleoperator, motor: str) -> None:
    bus = _require_single_bus(device)
    existing_calibration = device.calibration[motor]

    with bus.torque_disabled(motor):
        input(f"Move '{motor}' to the middle of its range of motion and press ENTER...")
        homing_offset = int(bus.set_half_turn_homings([motor])[motor])

        if _is_full_turn_motor(bus, motor, existing_calibration):
            range_min = existing_calibration.range_min
            range_max = existing_calibration.range_max
            print(f"Keeping full-turn range for '{motor}': {range_min} to {range_max}.")
        else:
            print(
                f"Move '{motor}' through its full range of motion.\n"
                "Recording positions. Press ENTER to stop..."
            )
            range_mins, range_maxes = bus.record_ranges_of_motion([motor])
            range_min = int(range_mins[motor])
            range_max = int(range_maxes[motor])

    updated_calibration = type(existing_calibration)(
        id=existing_calibration.id,
        drive_mode=existing_calibration.drive_mode,
        homing_offset=homing_offset,
        range_min=range_min,
        range_max=range_max,
    )

    device.calibration[motor] = updated_calibration
    bus.write_calibration({motor: updated_calibration}, cache=False)
    bus.calibration = device.calibration
    device._save_calibration()
    print(f"Updated '{motor}' in {device.calibration_fpath}")


def _choose_motor(device: Robot | Teleoperator) -> str:
    motors = list(device.calibration)
    if not motors:
        raise ValueError(f"No motors found in calibration file '{device.calibration_fpath}'.")

    if len(motors) == 1:
        chosen = motors[0]
        print(f"Using motor: {chosen}")
        return chosen

    print("\nAvailable motors:")
    labels = []
    for idx, motor in enumerate(motors, start=1):
        label = f"[{idx}] {motor}"
        labels.append(label)
        print(label)

    selected = _prompt_for_index("\nSelect a motor: ", labels)
    return motors[selected]


@draccus.wrap()
def recalibrate_motor(cfg: RecalibrateMotorConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    device = _make_device(cfg.device)
    calibration_fpath = _choose_calibration_file(device)
    _load_selected_calibration(device, calibration_fpath)
    motor = _choose_motor(device)

    device.connect(calibrate=False)
    try:
        recalibrate_selected_motor(device, motor)
    finally:
        device.disconnect()


def main():
    register_third_party_plugins()
    recalibrate_motor()


if __name__ == "__main__":
    main()
