"""Environment package for the inventory management RL framework.

This module exposes the available environment classes.
"""

# Use try/except to support both package and script modes
try:
    from .base_env import BaseInventoryEnv
    from .serial_env import SerialInventoryEnv
    from .network_env import NetworkInventoryEnv
    from .reward_functions import (
        holding_cost,
        backlog_cost,
        ordering_cost,
        bullwhip_effect,
        fill_rate,
        service_level,
    )
except ImportError:
    from envs.base_env import BaseInventoryEnv
    from envs.serial_env import SerialInventoryEnv
    from envs.network_env import NetworkInventoryEnv
    from envs.reward_functions import (
        holding_cost,
        backlog_cost,
        ordering_cost,
        bullwhip_effect,
        fill_rate,
        service_level,
    )

__all__ = [
    "BaseInventoryEnv",
    "SerialInventoryEnv",
    "NetworkInventoryEnv",
    "holding_cost",
    "backlog_cost",
    "ordering_cost",
    "bullwhip_effect",
    "fill_rate",
    "service_level",
]