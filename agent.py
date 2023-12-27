import random
from enum import Enum
from typing import List, Tuple

import numpy as np

from config import (
    NUMBER_OF_STATES,
    NUMBER_OF_POSSIBLE_ACTIONS,
    EPSILON_EXPLORATION,
    TERMINAL_PROBABILITY,
    AMOUNT_OF_UPDATES_PER_EPISODE,
    MAX_AMOUNT_UPDATES,
)
from env import Env, Action, State


class SamplingStrategy(Enum):
    COMPLETE_SWOOP = "complete_swoop"
    TRAJECTORY = "trajectory"


class Agent:
    def __init__(self, env: Env, sampling_strategy: SamplingStrategy):
        self.env = env
        self.q = np.zeros((NUMBER_OF_STATES, NUMBER_OF_POSSIBLE_ACTIONS))
        self.model: List[Tuple[State, Action]] = []
        self.sampling_strategy = sampling_strategy

    def update_model(self):
        amount_of_updates = 0
        values_of_v0 = [0]
        while amount_of_updates < MAX_AMOUNT_UPDATES:
            if self.sampling_strategy == SamplingStrategy.COMPLETE_SWOOP:
                for state in range(NUMBER_OF_STATES):
                    for action in range(NUMBER_OF_POSSIBLE_ACTIONS):
                        self.q[
                            (state, action)
                        ] = self._compute_expected_state_action_value(state, action)
                        amount_of_updates += 1
                        values_of_v0 += [np.max(self.q[0])]
                        if amount_of_updates == MAX_AMOUNT_UPDATES:
                            break
                        if amount_of_updates % 1000 == 0:
                            print(f"Update {amount_of_updates} done")
            elif self.sampling_strategy == SamplingStrategy.TRAJECTORY:
                self._generate_episode()
                for _ in range(AMOUNT_OF_UPDATES_PER_EPISODE):
                    state, action = random.choice(self.model)
                    self.q[(state, action)] = self._compute_expected_state_action_value(
                        state, action
                    )
                    amount_of_updates += 1
                    values_of_v0 += [np.max(self.q[0])]
                    if amount_of_updates == MAX_AMOUNT_UPDATES:
                        break
                    if amount_of_updates % 1000 == 0:
                        print(f"Update {amount_of_updates} done")
        return values_of_v0

    def _generate_episode(self):
        state = self.env.initial_state()
        terminal = False
        while not terminal:
            action = self._select_eps_greedy_action(state)
            self.model += [(state, action)]
            state, _, terminal = self.env.transition(state, action)

    def _compute_expected_state_action_value(self, state: State, action: Action):
        expected_value = 0
        probability_of_single_transition = (
            1 - TERMINAL_PROBABILITY
        ) / self.env.branching_factor
        for next_state, reward in self.env.transitions[state, action]:
            next_action = self._select_greedy_action(next_state)
            expected_value += probability_of_single_transition * (
                reward + self.q[(next_state, next_action)]
            )
        return expected_value

    def _select_eps_greedy_action(self, state: State):
        if random.random() > EPSILON_EXPLORATION:
            return self._select_greedy_action(state)
        return Action(random.choice(self.env.possible_actions()))

    def _select_greedy_action(self, state: State):
        return Action(np.argmax(self.q[state, :]))
