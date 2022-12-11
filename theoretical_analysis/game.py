from typing import List, Tuple, Union
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

    def _get_payoff_mitigate(self, player: int, N_all: Tuple[int], M_all: Tuple[int], t: int) -> float:
        '''
        :param player: Player to get payoff for
        :param N_all: Number of non-mitigation actions by each player up to this point
        :param M_all: Number of mitigation actions by player each player up to this point
        :param t: Time step (one time step is both players taking an action)
        '''
        N = sum(N_all)
        M = M_all[player]

        public_benefit = -self.d[player] * (N * np.log(N+1)) ** 2 # log(N+1) is to avoid log(0) and log(1) = 0 at N = 1
        private_cost = self.mu * 3 * self.k[player] * M * (self.gamma ** M)
        private_benefit = self.mu * self.b[player] * M * (1 + t / 10)

        print(public_benefit, private_cost, private_benefit)

        return public_benefit - private_cost + private_benefit + 3

    # def _get_payoff_no_mitigate(self, player: int, N: int, M: int, t: int) -> float:
    #     '''
    #     :param player: Player to get payoff for
    #     :param N: Number of non-mitigation actions up to this point
    #     :param M: Number of mitigation actions up to this point
    #     :param t: Time step (one time step is both players taking an action)
    #     '''
    #     public_benefit = -self.d[player] * (N * np.log(N)) ** 2

    #     return public_benefit

    def get_node(self, player: int, N_all: Tuple[int], M_all: Tuple[int], t: int) -> Union[Node, Payoff]:
        '''
        :param player: Player to get node for
        :param N_all: Number of non-mitigation actions by each player up to this point
        :param M_all: Number of mitigation actions by player each player up to this point
        :param t: Time step (one time step is both players taking an action)

        :returns: Node or Payoff
        '''
        next_player = 1 - player
        next_timestep = t if player == 0 else t + 1

        if t == self.max_timesteps and player == 0:
            return Payoff(a=self._get_payoff_mitigate(0, N_all, M_all, t), b=self._get_payoff_mitigate(1, N_all, M_all, t))

        else:
            if player == 0:
                mitigate_action = (M_all[0] + 1, M_all[1])
                non_mitigate_action = (N_all[0] + 1, N_all[1])
            else:
                mitigate_action = (M_all[0], M_all[1] + 1)
                non_mitigate_action = (N_all[0], N_all[1] + 1)

            return Node(
                info=(N_all, M_all, t),
                player=player,
                actions=(
                    self.get_node(next_player, N_all, mitigate_action, next_timestep), # mitigate
                    self.get_node(next_player, non_mitigate_action, M_all, next_timestep) # no mitigate
                )
            )

    def get_root(self):
        return self.get_node(0, (0, 0), (0, 0), 0)


if __name__ == "__main__":
    # simulate ToC stage game
    d = [0.5, 0.5]
    k = [0.9, 0.9]
    b = [0, 0]
    mu = 1
    game = Game(d, k, b, mu, max_timesteps=1)
    root = game.get_root()
    print(root)
    solver = Solver(root)
    solution = solver.solve()
    print(solution[0])
    print(solution[1])
    print(solution[2])

    print('====================')

    # simulate stage game of private benefits
    d = [0.5, 0.5]
    k = [0.9, 0.9]
    b = [0.1, 0.1]
    mu = 1
    game = Game(d, k, b, mu, max_timesteps=1)
    root = game.get_root()
    print(root)
    solver = Solver(root)
    solution = solver.solve()
    print(solution[0])
    print(solution[1])
    print(solution[2])