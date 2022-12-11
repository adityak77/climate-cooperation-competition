from typing import List, Union
from solver import Node, Payoff, Solver
import numpy as np

class Game:
    def __init__(self, d: List[float], k: List[float], b: List[float], mu: float, gamma: float = 0.87, max_timesteps: int = 20):
        '''
        :param d: List of values for player-specific public benefit of mitigation
        :param k: List of values for player-specific private cost of mitigation
        :param b: List of values for player-specific private benefit of mitigation
        :param gamma: private cost decay constant
        :param mu: mitigation value
        :param max_timesteps: maximum number of time steps to run the game for
        '''
        self.d = d
        self.k = k
        self.b = b
        self.gamma = gamma
        self.mu = mu
        self.max_timesteps = max_timesteps

    def _get_payoff_mitigate(self, player: int, N: int, M: int, t: int) -> float:
        '''
        :param player: Player to get payoff for
        :param N: Number of non-mitigation actions up to this point
        :param M: Number of mitigation actions up to this point
        :param t: Time step (one time step is both players taking an action)
        '''
        public_benefit = -self.d[player] * (N * np.log(N)) ** 2 if N > 0 else 0
        private_cost = self.mu * 3 * self.k[player] * (self.gamma ** M)
        private_benefit = self.mu * self.b[player] * (1 + t / 10)

        return public_benefit - private_cost + private_benefit + 3

    def _get_payoff_no_mitigate(self, player: int, N: int, M: int, t: int) -> float:
        '''
        :param player: Player to get payoff for
        :param N: Number of non-mitigation actions up to this point
        :param M: Number of mitigation actions up to this point
        :param t: Time step (one time step is both players taking an action)
        '''
        public_benefit = -self.d[player] * (N * np.log(N)) ** 2

        return public_benefit

    def get_node(self, player: int, N: int, M: int, t: int) -> Union[Node, Payoff]:
        '''
        :param player: Player to get node for
        :param N: Number of non-mitigation actions up to this point
        :param M: Number of mitigation actions up to this point
        :param t: Time step (one time step is both players taking an action)

        :returns: Node or Payoff
        '''
        next_player = 1 - player
        next_timestep = t if player == 0 else t + 1

        if t == self.max_timesteps:
            return Payoff(a=self._get_payoff_mitigate(0, N, M, t), b=self._get_payoff_mitigate(1, N, M, t))

        else:
            return Node(
                info=(N, M, t),
                player=player,
                actions=(
                    self.get_node(next_player, N, M + 1, next_timestep), # mitigate
                    self.get_node(next_player, N + 1, M, next_timestep) # no mitigate
                )
            )

    def get_root(self):
        return self.get_node(0, 0, 0, 0)


if __name__ == "__main__":
    d = [1, 1]
    k = [1, 1]
    b = [1, 1]
    mu = 1
    game = Game(d, k, b, mu, max_timesteps=5)
    solver = Solver(game.get_root())
    print(solver.solve())