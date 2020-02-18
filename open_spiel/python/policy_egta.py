# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Representation of a policy for a game.

This is a standard representation for passing policies into algorithms,
with currently the following implementations:

  TabularPolicy - an explicit policy per state, stored in an array
    of shape `(num_states, num_actions)`, convenient for tabular policy
    solution methods.
  UniformRandomPolicy - a uniform distribution over all legal actions for
    the specified player. This is computed as needed, so can be used for
    games where a tabular policy would be unfeasibly large.

The main way of using a policy is to call `action_probabilities(state,
player_id`), to obtain a dict of {action: probability}. `TabularPolicy`
objects expose a lower-level interface, which may be more efficient for
some use cases.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from open_spiel.python.algorithms import get_all_states
import pyspiel
from open_spiel.python.policy import Policy
from open_spiel.python import rl_agent, rl_environment


class UniformAgent(Policy):
  """Policy where the action distribution is uniform over all legal actions.

  This is computed as needed, so can be used for games where a tabular policy
  would be unfeasibly large, but incurs a legal action computation every time.
  """

  def __init__(self, game):
    """Initializes a uniform random policy for all players in the game."""
    all_players = list(range(game.num_players()))
    super(UniformAgent, self).__init__(game, all_players)
    # self._game = game
    self.env = rl_environment.Environment(game)

  def action_probabilities(self, state, player_id=None):
    legal_actions = (
        state.legal_actions()
        if player_id is None else state.legal_actions(player_id))
    probability = 1 / len(legal_actions)
    return {action: probability for action in legal_actions}

  def step(self, time_step, **kargs):
    player_id = time_step.observations["current_player"]
    legal_actions = time_step.observations["legal_actions"][player_id]

    num_legal_actions = len(legal_actions)
    num_actions = self.env.action_spec()["num_actions"]

    probs = np.zeros(num_actions)
    if num_legal_actions != 0:
      probs[legal_actions] = 1/num_legal_actions
    else:
      raise ValueError("The number of legal actions is zero.")

    action = np.random.choice(num_actions, p=probs)

    return rl_agent.StepOutput(action=action, probs=probs)


class RLPolcy(Policy):
  """
  This class provides a wrapper for RL agent so that a RL agent can be passed to certain functions,
  for example, a best response function.
  """
  #TODO: notice that the best response function cannot deal with mixed strategy.
  def __init__(self,
               game,
               agent,
               ):
    all_players = list(range(game.num_players()))
    super(RLPolcy, self).__init__(game, all_players)
    self._agent = agent
    self.env = rl_environment.Environment(game)

  def state_to_info_state(self, state):
    if self.game.get_type().provides_information_state_tensor:
      self._use_observation = False
    elif self.game.get_type().provides_observation_tensor:
      self._use_observation = True
    else:
      raise ValueError("Game must provide either information state or "
                       "observation as a normalized vector")

    player_id = state.current_player()
    if self._use_observation:
      info_state = state.observation_tensor(player_id)
    else:
      info_state = state.information_state_tensor(player_id)

    legal_actions = state.legal_actions(player_id)

    return info_state, legal_actions


  def action_probabilities(self, state, player_id=None):
    info_state, legal_actions = self.state_to_info_state(state)
    _, probs = self._agent._act(info_state, legal_actions)
    actions = range(len(probs))
    return dict(zip(actions,probs))
