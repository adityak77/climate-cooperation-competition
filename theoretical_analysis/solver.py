from collections import namedtuple
from dataclasses import dataclass
from typing import Union, Tuple, Hashable

Payoff = namedtuple("Payoff", ["a", "b"])


@dataclass(frozen=True)
class Node:
    # [info] should be used to store things like n, m, and t
    # for use later in move analysis
    info: Hashable
    player: int
    # [actions] is a tuple of nodes, where the ith element of the tuple represents
    # the ith action
    actions: Tuple[Union["Node", Payoff]]


class Solver:
    def __init__(self, game_root: Node):
        self.game_root = game_root
        self.strategy = None

    def _solve_subtree(self, subtree: Node):
        if isinstance(subtree, Payoff):
            return subtree, []

        subtree_sols = [(i, self._solve_subtree(action)) for i, action in enumerate(subtree.actions)]

        # [max] returns the first encountered item in case of a tie. We want to break ties in favor of
        # mitigation, so we reverse the subtree_sols
        best_action, (payoff, path) = max(subtree_sols[::-1], key=lambda x: x[1][0][subtree.player])

        self.strategy[subtree] = best_action

        return payoff, [(subtree, best_action), ] + path

    def solve(self):
        # Returns: payoff, strategy (mapping from Node -> Action), equilibrium path ((Node * Action) list)
        self.strategy = {}
        payoff, path = self._solve_subtree(self.game_root)

        return payoff, self.strategy, path


if __name__ == "__main__":
    # Run a test with a single Prisoner's Dilemma
    b_after_cooperate = Node(
        info="b_after_cooperate",
        player=1,
        actions=(Payoff(a=-2, b=-2), Payoff(a=-10, b=0))
    )

    b_after_defect = Node(
        info="b_after_defect",
        player=1,
        actions=(Payoff(a=0, b=-10), Payoff(a=-5, b=-5))
    )

    root = Node(
        info="root",
        player=0,
        actions=(b_after_cooperate, b_after_defect)
    )

    solver = Solver(root)
    print(solver.solve())

