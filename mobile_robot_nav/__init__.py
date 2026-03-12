# SPDX-License-Identifier: MIT
"""Public API for mobile_robot_nav."""

from gymnasium.envs.registration import register

from .__version__ import __version__
from .envs import MobileRobotNavEnv

register(
    id="MobileRobotNav-v0",
    entry_point="mobile_robot_nav.envs:MobileRobotNavEnv",
)

__all__ = [
    "MobileRobotNavEnv",
    "__version__",
]

__author__ = "Md Shahriar Forhad"
__email__ = "shahriar.forhad.eee@gmail.com"
__license__ = "MIT"