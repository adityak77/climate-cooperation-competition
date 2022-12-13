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

        return public_benefit - private_cost + private_benefit + 3

    def _get_stage_payoff(self, actions: Tuple[int], N_all: Tuple[int], M_all: Tuple[int], t: int) -> float:
        N = sum(N_all)
        M = sum(M_all)

        player = 0

        # public_benefit = -self.d[player] * (N * np.log(N+1)) ** 2
        public_benefit = -self.d[player] * N
        private_cost = (1-actions[player]) * 3 * self.k[player] * (self.gamma ** M)
        private_benefit = (1-actions[player]) * self.b[player] * (1 + t / 10)

        p0 = public_benefit - private_cost + private_benefit

        player = 1

        # public_benefit = -self.d[player] * (N * np.log(N+1)) ** 2
        public_benefit = -self.d[player] * N
        private_cost = (1-actions[player]) * 3 * self.k[player] * (self.gamma ** M)
        private_benefit = (1-actions[player]) * self.b[player] * (1 + t / 10)

        p1 = public_benefit - private_cost + private_benefit

        return Payoff(a=p0, b=p1)

    def _get_stage_game(self, info):
        N_all, M_all, t = info

        actions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        payoffs = []

        for aa, ab in actions:
            # N = (N_all[0] + (aa == 1), N_all[1] + (ab == 1))
            # M = (M_all[0] + (aa == 0), M_all[1] + (ab == 0))
            payoff = self._get_stage_payoff((aa, ab), N_all, M_all, t)
            payoffs.append(payoff)

        return payoffs

    # def _get_payoff_no_mitigate(self, player: int, N: int, M: int, t: int) -> float:
    #     '''
    #     :param player: Player to get payoff for
    #     :param N: Number of non-mitigation actions up to this point
    #     :param M: Number of mitigation actions up to this point
    #     :param t: Time step (one time step is both players taking an action)
    #     '''
    #     public_benefit = -self.d[player] * (N * np.log(N)) ** 2

    #     return public_benefit

    def get_node(self, player: int, N_all: Tuple[int], M_all: Tuple[int], t: int, curr_cost=Payoff(a=0, b=0), last_action=-1) -> Union[Node, Payoff]:
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
            # return Payoff(a=self._get_payoff_mitigate(0, N_all, M_all, t), b=self._get_payoff_mitigate(1, N_all, M_all, t))
            return curr_cost

        else:
            if player == 0:
                mitigate_action = (M_all[0] + 1, M_all[1])
                non_mitigate_action = (N_all[0] + 1, N_all[1])
                m_payoff = curr_cost
                n_payoff = curr_cost
            else:
                mitigate_action = (M_all[0], M_all[1] + 1)
                non_mitigate_action = (N_all[0], N_all[1] + 1)
                prev_N = (N_all[0] - last_action, N_all[1])
                prev_M = (M_all[0] - 1 + last_action, M_all[1])
                m_delta_payoff = self._get_stage_payoff((last_action, 0), prev_N, prev_M, t)
                n_delta_payoff = self._get_stage_payoff((last_action, 1), prev_N, prev_M, t)
                m_payoff = Payoff(a=curr_cost.a+m_delta_payoff.a, b=curr_cost.b+m_delta_payoff.b)
                n_payoff = Payoff(a=curr_cost.a+n_delta_payoff.a, b=curr_cost.b+n_delta_payoff.b)

            return Node(
                info=(N_all, M_all, t),
                player=player,
                actions=(
                    self.get_node(next_player, N_all, mitigate_action, next_timestep, curr_cost=m_payoff, last_action=0), # mitigate
                    self.get_node(next_player, non_mitigate_action, M_all, next_timestep, curr_cost=n_payoff, last_action=1) # no mitigate
                )
            )

    def get_root(self):
        return self.get_node(0, (0, 0), (0, 0), 0)

def payoff_all_mitigate(node):
    if isinstance(node, Payoff):
        return node

    return payoff_all_mitigate(node.actions[0])

def payoff_all_non(node):
    if isinstance(node, Payoff):
        return node

    return payoff_all_non(node.actions[1])

def find_eq_path(node, strategy):
    if isinstance(node, Payoff):
        return []

    action = strategy[node]
    return [action] + find_eq_path(node.actions[action], strategy)

if __name__ == "__main__":
    # simulate ToC stage game
    d = [0.5, 0.5]
    k = [0.9, 0.9]
    b = [0, 0]
    mu = 1
    game = Game(d, k, b, mu, max_timesteps=1)
    root = game.get_root()

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

    solver = Solver(root)
    solution = solver.solve()
    print(solution[0])
    print(solution[1])
    print(solution[2])

    print('====================')

    # simulate private benefits + multiple time steps (includes increasing returns)
    d = [0.5, 0.5]
    k = [0.9, 0.9]
    b = [0.1, 0.1]
    mu = 1
    game = Game(d, k, b, mu, max_timesteps=5)
    root = game.get_root()
    # print(root)
    solver = Solver(root)
    solution = solver.solve()
    
    fraction_mitigate = sum([1 - elem[1] for elem in solution[2]]) / len(solution[2])
    
    print('Equilibrium path actions', [elem[1] for elem in solution[2]])
    print(fraction_mitigate)
    
    print('====================')

    # simulate private benefits + multiple time steps (includes increasing returns)
    d = [0.72, 0.72]
    k = [1.0, 1.0]
    b = [0.0, 0.0]
    mu = 1
    game = Game(d, k, b, mu, max_timesteps=5)
    root = game.get_root()
    # print(root)
    solver = Solver(root)
    solution = solver.solve()
    
    fraction_mitigate = sum([1 - elem[1] for elem in solution[2]]) / len(solution[2])
    
    print('Equilibrium path actions', [elem[1] for elem in solution[2]])
    print(fraction_mitigate)
    print(game._get_stage_game(solution[2][1][0].info))
    print(f"mitigate: {payoff_all_mitigate(root)}, non: {payoff_all_non(root)}")

    print('====================')

    # simulate private benefits + multiple time steps (includes increasing returns)
    d = [1., 0.72]
    k = [0.6, 1.0]
    b = [0.2, 0.0]
    mu = 1
    game = Game(d, k, b, mu, max_timesteps=5)
    root = game.get_root()
    # print(root)
    solver = Solver(root)
    solution = solver.solve()
    
    fraction_mitigate = sum([1 - elem[1] for elem in solution[2]]) / len(solution[2])
    
    print('Equilibrium path actions', [elem[1] for elem in solution[2]])
    print(fraction_mitigate)
    print(game._get_stage_game(solution[2][0][0].info))
    print(game._get_stage_game(solution[2][-4][0].info))
    print(f"mitigate: {payoff_all_mitigate(root)}, non: {payoff_all_non(root)}")
    print(find_eq_path(solution[2][1][0].actions[1], solution[1]))
    print(find_eq_path(solution[2][1][0].actions[0], solution[1]))

    print('====================')

    # simulate private benefits + multiple time steps (includes increasing returns)
    d = [1., 0.7]
    k = [0.6, 1.0]
    b = [0.3, 0.2]
    mu = 1
    game = Game(d, k, b, mu, max_timesteps=5)
    root = game.get_root()
    # print(root)
    solver = Solver(root)
    solution = solver.solve()
    
    fraction_mitigate = sum([1 - elem[1] for elem in solution[2]]) / len(solution[2])
    
    print('Equilibrium path actions', [elem[1] for elem in solution[2]])
    print(fraction_mitigate)
    print(game._get_stage_game(solution[2][-2][0].info))
    print(game._get_stage_game(solution[2][-4][0].info))
    print(f"mitigate: {payoff_all_mitigate(root)}, non: {payoff_all_non(root)}")
    print(find_eq_path(solution[2][1][0].actions[1], solution[1]))
    print(find_eq_path(solution[2][1][0].actions[0], solution[1]))

    print('====================')

    # simulate private benefits + multiple time steps (includes increasing returns)
    d = [0.72, 0.72]
    k = [1.0, 1.0]
    b = [0.0, 0.0]
    mu = 1
    game = Game(d, k, b, mu, gamma=1, max_timesteps=5)
    root = game.get_root()
    # print(root)
    solver = Solver(root)
    solution = solver.solve()
    
    fraction_mitigate = sum([1 - elem[1] for elem in solution[2]]) / len(solution[2])
    
    print('Equilibrium path actions', [elem[1] for elem in solution[2]])
    print(fraction_mitigate)
    print(game._get_stage_game(solution[2][-1][0].info))
    print(f"mitigate: {payoff_all_mitigate(root)}, non: {payoff_all_non(root)}")
    print(find_eq_path(solution[2][1][0].actions[1], solution[1]))
