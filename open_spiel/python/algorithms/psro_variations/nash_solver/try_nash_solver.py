from open_spiel.python.algorithms.psro_variations.nash_solver import general_nash_solver as gs

import numpy as np

# Games

# (1) Matching Pennies
MP_p1_meta_game = np.array([[1, -1], [-1, 1]])
MP_p2_meta_game = np.array([[-1, 1], [1, -1]])
MP_meta_games = [MP_p1_meta_game, MP_p2_meta_game]

#(2) Battle of Sexes
BOS_p1_meta_game = np.array([[3, 0], [0, 2]])
BOS_p2_meta_game = np.array([[2, 0], [0, 3]])
BOS_meta_games = [BOS_p1_meta_game, BOS_p2_meta_game]

#(3) Bar Crowding Game (3 players)
BC_p1_meta_game = np.array([[[-1, 2],[1, 1]], [[2, 0], [1, 1]]])
BC_p2_meta_game = np.array([[[-1, 1],[2, 1]], [[2, 1], [0, 1]]])
BC_p3_meta_game = np.array([[[-1, 2],[2, 0]], [[1, 1], [1, 1]]])
BC_meta_games = [BC_p1_meta_game, BC_p2_meta_game, BC_p3_meta_game]

game_name = 'BOS'

if game_name == 'MP':
    meta_games = MP_meta_games
elif game_name == 'BOS':
    meta_games = BOS_meta_games
elif game_name == 'BC':
    meta_games = BC_meta_games
else:
    raise ValueError("Game does not exist.")

# gt.encode_gambit_file(meta_games)
# gt.gambit_analysis(600)
# equilibria = gt.decode_gambit_file(meta_games, mode="all")
# for eq in equilibria:
#     print(eq)

# equilibria = gt.do_gambit_analysis(meta_games, mode="pure", timeout = 600)
# for eq in equilibria:
#     print(eq)

equilibria = gs.nash_solver(meta_games, solver="gambit", mode='all')
print(equilibria)
for eq in equilibria:
    print(eq)

# tol = 1e-6
# row_payoffs, col_payoffs = meta_games[0], meta_games[1]
# pure_nash = list(
#   zip(*((row_payoffs >= row_payoffs.max(0, keepdims=True) - tol)
#         & (col_payoffs >= col_payoffs.max(1, keepdims=True) - tol)
#        ).nonzero()))
# print(pure_nash)