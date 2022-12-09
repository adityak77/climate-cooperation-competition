from collections import namedtuple
from dataclasses import dataclass
from typing import Union, Dict

Action = namedtuple("Action", ["a", "b"])
Payoff = namedtuple("Payoff", ["a", "b"])

@dataclass
class Node:
    actions: Dict[Action, Union["Node", Payoff]]
    is_leaf: bool

class Solver:
    def __init__(self, game_root: Node):
        self.game_root = None

    def solve(self):
        # Returns: strategy (mapping from Node -> Action), equilibrium path (Action list)
        raise NotImplementedError
