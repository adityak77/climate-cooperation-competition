from collections import namedtuple
from dataclasses import dataclass
from typing import Union, Dict, Hashable

Payoff = namedtuple("Payoff", ["a", "b"])

@dataclass
class Node:
    # [info] should be used to store things like n, m, and t
    # for use later in move analysis
    info: Hashable
    player: int
    actions: Dict[int, Union["Node", Payoff]]

class Solver:
    def __init__(self, game_root: Node):
        self.game_root = game_root
        self.strategy = None

    def _solve_subtree(self, subtree: Node):
        if isinstance(subtree, Payoff):
            return subtree, []

        num_actions = len(subtree.actions)
        subtree_sols = [(act, self._solve_subtree(subtree.actions[act])) for act in range(num_actions)]

        # [max] returns the first encountered item in case of a tie. We want to break ties in favor of
        # mitigation, so we reverse the subtree_sols
        best_action, (payoff, path) = max(subtree_sols[::-1], key=lambda x: x[1][0][subtree.player])

        self.strategy[subtree] = best_action

        return payoff, [(subtree, best_action), ] + path

    def solve(self):
        # Returns: strategy (mapping from Node -> Action), equilibrium path ((Node * Action) list)
        self.strategy = {}
        _, path = self._solve_subtree(self.game_root)

        return self.strategy, path

