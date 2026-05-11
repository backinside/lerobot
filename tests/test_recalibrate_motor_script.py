from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from lerobot.motors import MotorCalibration
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.scripts.lerobot_recalibrate_motor import (
    _load_selected_calibration,
    discover_calibration_files,
    recalibrate_selected_motor,
)


@dataclass(kw_only=True)
class StubRobotConfig(RobotConfig):
    port: str = "/dev/null"


class FakeBus:
    def __init__(self):
        self.calibration = {}
        self.motors = {
            "joint_a": SimpleNamespace(model="sts3215"),
            "joint_b": SimpleNamespace(model="sts3215"),
        }
        self.model_resolution_table = {"sts3215": 4096}
        self.write_calls = []
        self.record_calls = []

    @contextmanager
    def torque_disabled(self, motors=None):
        yield

    def set_half_turn_homings(self, motors=None):
        target = motors[0]
        return {target: 123}

    def record_ranges_of_motion(self, motors=None):
        target = motors[0]
        self.record_calls.append(target)
        return {target: 111}, {target: 222}

    def write_calibration(self, calibration_dict, cache=True):
        self.write_calls.append((calibration_dict, cache))
        if cache:
            self.calibration = calibration_dict


class StubRobot(Robot):
    config_class = StubRobotConfig
    name = "stub_robot"
    legacy_calibration_names = ("legacy_stub_robot",)

    def __init__(self, config: StubRobotConfig):
        super().__init__(config)
        self.bus = FakeBus()

    @property
    def observation_features(self) -> dict:
        return {}

    @property
    def action_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return False

    def connect(self, calibrate: bool = True) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self):
        return {}

    def send_action(self, action):
        return action

    def disconnect(self) -> None:
        pass


def test_discover_calibration_files_includes_legacy(monkeypatch, tmp_path):
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    legacy_dir = tmp_path / "robots" / "legacy_stub_robot"
    legacy_dir.mkdir(parents=True)

    current_file = current_dir / "blue.json"
    legacy_file = legacy_dir / "green.json"
    duplicate_legacy_name = legacy_dir / "blue.json"
    current_file.write_text("{}")
    legacy_file.write_text("{}")
    duplicate_legacy_name.write_text("{}")

    monkeypatch.setattr(
        "lerobot.scripts.lerobot_recalibrate_motor.HF_LEROBOT_CALIBRATION",
        tmp_path,
    )

    robot = StubRobot(StubRobotConfig(id="blue", calibration_dir=current_dir))

    calibration_files = discover_calibration_files(robot)

    assert calibration_files == sorted([current_file, duplicate_legacy_name, legacy_file])


def test_recalibrate_selected_motor_updates_only_target(monkeypatch, tmp_path):
    calibration_dir = tmp_path / "current"
    calibration_dir.mkdir()
    calibration_fpath = calibration_dir / "blue.json"

    robot = StubRobot(StubRobotConfig(id="blue", calibration_dir=calibration_dir))
    robot.calibration = {
        "joint_a": MotorCalibration(id=1, drive_mode=0, homing_offset=10, range_min=20, range_max=30),
        "joint_b": MotorCalibration(id=2, drive_mode=1, homing_offset=40, range_min=50, range_max=60),
    }
    robot._save_calibration(calibration_fpath)

    _load_selected_calibration(robot, calibration_fpath)

    monkeypatch.setattr("builtins.input", lambda _: "")

    recalibrate_selected_motor(robot, "joint_a")

    assert robot.calibration["joint_a"] == MotorCalibration(
        id=1,
        drive_mode=0,
        homing_offset=123,
        range_min=111,
        range_max=222,
    )
    assert robot.calibration["joint_b"] == MotorCalibration(
        id=2,
        drive_mode=1,
        homing_offset=40,
        range_min=50,
        range_max=60,
    )
    assert robot.bus.write_calls == [
        (
            {
                "joint_a": MotorCalibration(
                    id=1,
                    drive_mode=0,
                    homing_offset=123,
                    range_min=111,
                    range_max=222,
                )
            },
            False,
        )
    ]
    assert robot.bus.record_calls == ["joint_a"]

    reloaded = StubRobot(StubRobotConfig(id="blue", calibration_dir=calibration_dir))
    _load_selected_calibration(reloaded, calibration_fpath)
    assert reloaded.calibration == robot.calibration
