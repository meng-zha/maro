from dataclasses import dataclass
from typing import List

import numpy as np

from maro.rl_v3.utils import SHAPE_CHECK_FLAG


@dataclass
class TransitionBatch:
    states: np.ndarray  # 2D
    actions: np.ndarray  # 2D
    rewards: np.ndarray  # 1D
    terminals: np.ndarray  # 1D
    next_states: np.ndarray = None  # 2D
    values: np.ndarray = None  # 1D
    logps: np.ndarray = None  # 1D

    def __post_init__(self) -> None:
        if SHAPE_CHECK_FLAG:
            assert len(self.states.shape) == 2 and self.states.shape[0] > 0

            assert len(self.actions.shape) == 2 and self.actions.shape[0] == self.states.shape[0]
            assert len(self.rewards.shape) == 1 and self.rewards.shape[0] == self.states.shape[0]

            assert len(self.terminals.shape) == 1 and self.terminals.shape[0] == self.states.shape[0]
            assert self.next_states is None or self.next_states.shape == self.states.shape
            assert self.values is None or self.values.shape == self.terminals.shape
            assert self.logps is None or self.logps.shape == self.terminals.shape


@dataclass
class MultiTransitionBatch:
    states: np.ndarray  # 2D
    actions: List[np.ndarray]  # 2D
    rewards: List[np.ndarray]  # 1D
    terminals: np.ndarray  # 1D
    local_states: List[np.ndarray] = None  # 2D
    next_states: np.ndarray = None  # 2D
    values: np.ndarray = None  # 1D
    logps: np.ndarray = None  # 1D

    def __post_init__(self) -> None:
        if SHAPE_CHECK_FLAG:
            assert len(self.states.shape) == 2 and self.states.shape[0] > 0

            assert len(self.actions) == len(self.rewards)
            assert self.local_states is None or len(self.local_states) == len(self.actions)
            for i in range(len(self.actions)):
                assert len(self.actions[i].shape) == 2 and self.actions[i].shape[0] == self.states.shape[0]
                assert len(self.rewards[i].shape) == 1 and self.rewards[i].shape[0] == self.states.shape[0]
                if self.local_states is not None:
                    assert len(self.local_states[i].shape) == 2
                    assert self.local_states[i].shape[0] == self.states.shape[0]

            assert len(self.terminals.shape) == 1 and self.terminals.shape[0] == self.states.shape[0]
            assert self.next_states is None or self.next_states.shape == self.states.shape
            assert self.values is None or self.values.shape == self.terminals.shape
            assert self.logps is None or self.logps.shape == self.terminals.shape
