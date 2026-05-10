# !/usr/bin/env python

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

import time

from lerobot.robots.so_follower import SO101FollowerClient, SO101FollowerClientConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30


def main():
    # Update these values for your setup.
    robot_config = SO101FollowerClientConfig(remote_ip="10.2.0.210", id="remote_so101")
    leader_config = SO101LeaderConfig(port="/dev/ttyACM0", id="local_so101_leader")

    robot = SO101FollowerClient(robot_config)
    leader = SO101Leader(leader_config)

    # Start the remote host first on the robot machine.
    robot.connect()
    leader.connect()

    init_rerun(session_name="so101_remote_teleop")

    try:
        print("Starting remote SO-101 teleop loop...")
        while True:
            t0 = time.perf_counter()

            observation = robot.get_observation()
            action = leader.get_action()
            _ = robot.send_action(action)

            log_rerun_data(observation=observation, action=action)

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    finally:
        leader.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
