from __future__ import annotations

import abc
from typing import List, Tuple, Any
import numpy as np


class BaseInventoryEnv(abc.ABC):
    """Abstract base class for multi–echelon inventory environments.

    Subclasses must implement the core RL interface methods: :meth:`reset`
    and :meth:`step`.  They should also set the attributes ``agent_num`` and
    ``obs_dim`` to describe the number of agents (supply chain levels) and
    the dimensionality of each agent's observation vector.
    """

    #: Number of agents (echelons) in the environment.  Subclasses should
    #: override this in their ``__init__``.
    agent_num: int = 0
    #: Dimension of each agent's observation vector.  Should be set in
    #: subclasses once the observation structure is known.
    obs_dim: int = 0

    @abc.abstractmethod
    def reset(self, train: bool = True, normalize: bool = True) -> List[np.ndarray]:
        """Reset the environment to a starting state and return initial observations.

        Args:
            train: Whether the reset is for training (True) or evaluation (False).
            normalize: If True, normalise observation values to the range [0, 1].

        Returns:
            A list of observation arrays, one per agent.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, actions: List[int], one_hot: bool = True) -> Tuple[List[np.ndarray], List[List[float]], List[bool], List[Any]]:
        """Advance the environment by one step given actions from all agents.

        Args:
            actions: A list of actions from each agent.  Each element may be
                either an integer index into the action space or a one–hot
                encoded vector.  The ``one_hot`` flag specifies the form.
            one_hot: If True, interpret each element of ``actions`` as a
                one–hot vector and convert to an integer index via ``argmax``.

        Returns:
            Tuple ``(obs, rewards, done, info)`` as defined by RL interface:

            * ``obs`` – list of next observation arrays for each agent.
            * ``rewards`` – list of 1‐element lists containing the reward for
              each agent (this shape is expected by certain MARL frameworks).
            * ``done`` – list of booleans indicating whether the episode is
              finished for each agent.  Environments typically set all
              elements equal to the same flag.
            * ``info`` – list of optional diagnostic dictionaries (unused by
              most algorithms but provided for completeness).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional helper methods
    # ------------------------------------------------------------------
    def get_eval_num(self) -> int:
        """Return the number of evaluation demand sequences loaded.

        Subclasses should override this if they support evaluation data.
        """
        return 0

    def get_eval_bw_res(self) -> List[float]:
        """Return the bullwhip effect results accumulated during evaluation.

        Subclasses may override this to provide a list of bullwhip metrics
        after evaluation episodes.  The base implementation returns an empty
        list.
        """
        return []

    def get_orders(self) -> List[int]:
        """Return the most recent actual orders placed by each agent.

        Subclasses may override this to expose recent actions for logging.
        The base implementation returns an empty list.
        """
        return []

    def get_inventory(self) -> List[int]:
        """Return the current inventory level for each agent.

        Subclasses may override this to provide access to underlying state.
        The base implementation returns an empty list.
        """
        return []

    # --- FIX: ADDED CLOSE METHOD ---
    def close(self):
        """Override close method to clean up resources.
        
        This method is required by SubprocVecEnv to allow worker processes 
        to shut down gracefully.
        """
        pass
    # -------------------------------