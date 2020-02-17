'''
Author: Yongzhao Wang
Strategic Reasoning Group
University of Michigan
'''

import nashpy as nash
import numpy as np


def nash_strategy(solver, return_all=False):
  """Returns nash distribution on meta game matrix.
  This method only works for two player general-sum games.
  Args:
    solver: GenPSROSolver instance.
    return_all: if return all NE or random one.
  Returns:
    Nash distribution on strategies.
  """
  meta_games = solver.get_meta_game
  if not isinstance(meta_games, list):
    meta_games = [meta_games, -meta_games]
  if len(meta_games) > 2:
      raise ValueError("Nash solver only works for two player general-sum games. Number of players > 2.")

  p1_payoff = meta_games[0]
  p2_payoff = meta_games[1]
  game = nash.Game(p1_payoff, p2_payoff)
  NE_list = []
  for eq in game.support_enumeration():
      NE_list.append(eq)

  if return_all:
    return NE_list
  else:
    return list(np.random.choice(NE_list))



