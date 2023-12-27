import random
from typing import Dict, Tuple, List, Optional
import numpy as np
from config import (
    TERMINAL_PROBABILITY,
    NUMBER_OF_STATES,
    NUMBER_OF_POSSIBLE_ACTIONS,
    BRANCHING_FACTOR,
)

State = int
Action = int
Reward = float


class Env:
    def __init__(
        self,
        branching_factor: Optional[int] = BRANCHING_FACTOR,
        number_of_states: Optional[int] = NUMBER_OF_STATES,
        number_of_actions: Optional[int] = NUMBER_OF_POSSIBLE_ACTIONS,
    ):
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.branching_factor = branching_factor
        self.transitions: Dict[
            Tuple[State, Action], List[Tuple[State, Reward]]
        ] = self._create_transitions()

    def _create_transitions(self):
        transitions = {}
        for state in range(self.number_of_states):
            for action in range(self.number_of_actions):
                next_state_candidates = random.sample(
                    range(self.number_of_states), self.branching_factor
                )
                transitions[state, action] = [
                    (next_state_candidate, self._generate_reward())
                    for next_state_candidate in next_state_candidates
                ]
        return transitions

    @staticmethod
    def _generate_reward() -> float:
        return np.random.normal()

    def transition(self, state: State, action: Action) -> Tuple[State, Reward, bool]:
        if random.random() < TERMINAL_PROBABILITY:
            return -1, 0, True
        return (*random.choice(self.transitions[state, action]), False)

    @staticmethod
    def initial_state():
        return State(0)

    def possible_actions(self) -> List[Action]:
        return list(range(self.number_of_actions))
