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

"""Modular implementations of the PSRO meta algorithm with reinforcement learning oracle.

Allows the use of Restricted Nash Response, Nash Response, Uniform Response,
and other modular matchmaking selection components users can add.

This version works for N player, general sum games.

One iteration of the algorithm consists in :
1) Compute the selection probability vector for current list of strategies
2) From every strategy used (For which selection probability > 0 if training is
restricted), generate a new best response strategy against the
decision-probability-weighted mixture of strategies using an oracle ; perhaps
only considering agents in the mixture that are beaten (Rectified training
setting).
3) Update meta game matrix with new game results.


In this version, we combine RL with PSRO and initial strategies are uniform strategy.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import copy
import pickle
import pdb

from open_spiel.python.algorithms.psro_variations import abstract_meta_trainer_egta
from open_spiel.python.algorithms import exploitability
from open_spiel.python.policy_egta import UniformAgent,RLPolicy
from open_spiel.python.algorithms import policy_aggregator
from open_spiel.python import rl_environment

# Constant, specifying the threshold below which probabilities are considered
# 0 in the Rectified Nash Response setting.
EPSILON_MIN_POSITIVE_PROBA = 1e-6


def rectified_strategy_selector(solver):
  """Returns every strategy with nonzero selection probability.

  Args:
    solver: A GenPSROSolver instance.
  """
  used_policies = []
  policies = solver.get_policies
  num_players = len(policies)
  meta_strategy_probabilities = solver.get_and_update_meta_strategies(
      update=False)
  for k in range(num_players):
    current_policies = policies[k]
    current_probabilities = meta_strategy_probabilities[k]
    current_policies = [
        current_policies[i]
        for i in range(len(current_policies))
        if current_probabilities[i] > EPSILON_MIN_POSITIVE_PROBA
    ]
    used_policies.append(current_policies)
  return used_policies


def exhaustive_strategy_selector(solver):
  """Returns every player's policies.

  Args:
    solver: A GenPSROSolver instance.
  """
  return solver.get_policies


def probabilistic_strategy_selector(solver):
  """Returns [kwargs] policies randomly, proportionally with selection probas.

  Args:
    solver: A GenPSROSolver instance.
  """
  kwargs = solver.get_kwargs
  # By default, select only 1 new policy to optimize from.
  number_policies_to_select = kwargs.get("number_policies_selected") or 1
  policies = solver.get_policies
  num_players = len(policies)
  meta_strategy_probabilities = solver.get_and_update_meta_strategies(
      update=False)

  used_policies = []
  for k in range(num_players):
    current_policies = policies[k]
    current_selection_probabilities = meta_strategy_probabilities[k]
    effective_number = min(number_policies_to_select, len(current_policies))
    selected_policies = list(
        np.random.choice(
            current_policies,
            effective_number,
            replace=False,
            p=current_selection_probabilities))
    used_policies.append(selected_policies)
  return used_policies


def probabilistic_deterministic_strategy_selector(solver):
  """Returns [kwargs] policies with highest selection probabilities.

  Args:
    solver: A GenPSROSolver instance.
  """
  kwargs = solver.get_kwargs
  # By default, select only 1 new policy to optimize from.
  number_policies_to_select = kwargs.get("number_policies_selected") or 1
  policies = solver.get_policies
  num_players = len(policies)
  meta_strategy_probabilities = solver.get_and_update_meta_strategies(
      update=False)

  used_policies = []
  for k in range(num_players):
    current_policies = policies[k]
    current_selection_probabilities = meta_strategy_probabilities[k]
    effective_number = min(number_policies_to_select, len(current_policies))
    # pylint: disable=g-complex-comprehension
    selected_policies = [
        policy for _, policy in sorted(
            zip(current_selection_probabilities, current_policies),
            reverse=True,
            key=lambda pair: pair[0])
    ][:effective_number]
    used_policies.append(selected_policies)
  return used_policies


def uniform_strategy_selector(solver):
  """Returns [kwargs] randomly selected policies (Uniform probability).

  Args:
    solver: A GenPSROSolver instance.
  """
  kwargs = solver.get_kwargs
  # By default, select only 1 new policy to optimize from.
  number_policies_to_select = kwargs.get("number_policies_selected") or 1
  policies = solver.get_policies
  num_players = len(policies)

  used_policies = []
  for k in range(num_players):
    current_policies = policies[k]
    effective_number = min(number_policies_to_select, len(current_policies))
    selected_policies = list(
        np.random.choice(
            current_policies,
            effective_number,
            replace=False,
            p=np.ones(len(current_policies)) / len(current_policies)))
    used_policies.append(selected_policies)
  return used_policies


def functional_probabilistic_strategy_selector(solver):
  """Returns [kwargs] randomly selected policies with generated probabilities.

  Args:
    solver: A GenPSROSolver instance.
  """
  kwargs = solver.get_kwargs
  # By default, select only 1 new policy to optimize from.
  number_policies_to_select = kwargs.get("number_policies_selected") or 1
  # By default, use meta strategies.
  probability_computation_function = kwargs.get(
      "selection_probability_function") or (
          lambda x: x.get_and_update_meta_strategies(update=False))

  policies = solver.get_policies
  num_players = len(policies)

  meta_strategy_probabilities = probability_computation_function(solver)

  used_policies = []
  for k in range(num_players):
    current_policies = policies[k]
    current_selection_probabilities = meta_strategy_probabilities[k]
    effective_number = min(number_policies_to_select, len(current_policies))
    selected_policies = list(
        np.random.choice(
            current_policies,
            effective_number,
            replace=False,
            p=current_selection_probabilities))
    used_policies.append(selected_policies)
  return used_policies


TRAINING_STRATEGY_SELECTORS = {
    "probabilistic_deterministic":
        probabilistic_deterministic_strategy_selector,
    "functional_probabilistic":
        functional_probabilistic_strategy_selector,
    "probabilistic":
        probabilistic_strategy_selector,
    "exhaustive":
        exhaustive_strategy_selector,
    "rectified":
        rectified_strategy_selector,
    "uniform":
        uniform_strategy_selector
}

DEFAULT_STRATEGY_SELECTION_METHOD = "probabilistic"


def empty_list_generator(number_dimensions):
  result = []
  for _ in range(number_dimensions - 1):
    result = [result]
  return result


class GenEGTASolver(abstract_meta_trainer_egta.AbstractMetaTrainer):
  """A general implementation PSRO with reinforcement learning oracle.

  PSRO is the algorithm described in (Lanctot et Al., 2017,
  https://arxiv.org/pdf/1711.00832.pdf ).

  Subsequent work regarding PSRO's matchmaking and training has been performed
  by David Balduzzi, who introduced Restricted Nash Response (RNR), Nash
  Response (NR) and Uniform Response (UR).
  RNR is Algorithm 4 in (Balduzzi, 2019, "Open-ended Learning in Symmetric
  Zero-sum Games"). NR, Nash response, is algorithm 3.
  Balduzzi et Al., 2019, https://arxiv.org/pdf/1901.08106.pdf

  This implementation allows one to modularly choose different meta strategy
  computation methods, or other user-written ones.
  """

  def __init__(self,
               game,
               oracle,
               session,
               sims_per_entry,
               initial_policies=None,
               training_strategy_selector=None,
               meta_strategy_method='nash',
               symmetric=False,
               **kwargs):
    """Initialize the RNR solver.

    Arguments:
      game: The open_spiel game object.
      oracle: Callable that takes as input: - game - policy - policies played -
        array representing the probability of playing policy i - other kwargs
        and returns a new best response.
      sims_per_entry: Number of simulations to run to estimate each element of
        the game outcome matrix.
      initial_policies: A list of initial policies for each player, from which
        the optimization process will start.
      training_strategy_selector: Callable taking a GenPSROSolver object and
        returning a list of list of selected strategies to train from (One list
        entry per player), or string selecting pre-implemented methods.
        String value can be:
              - "probabilistic_deterministic": selects the first
                kwargs["number_policies_selected"] policies with highest
                selection probabilities.
              - "probabilistic": randomly selects
                kwargs["number_policies_selected"] with probabilities determined
                by the meta strategies.
              - "exhaustive": selects every policy of every player.
              - "rectified": only selects strategies that have nonzero chance of
                being selected.
              - "uniform": randomly selects kwargs["number_policies_selected"]
                policies with uniform probabilities.
      meta_strategy_method: String or callable taking a GenPSROSolver object and
        returning a list of meta strategies (One list entry per player).
        String value can be:
              - "uniform": Uniform distribution on policies.
              - "nash": Taking nash distribution. Only works for 2 player, 0-sum
                games.
              - "prd": Projected Replicator Dynamics, as described in Lanctot et
                Al.
      symmetric: if the game is symmetric or not.
      **kwargs: kwargs for meta strategy computation and training strategy
        selection.
    """
    self._sims_per_entry = sims_per_entry
    self._session = session
    self.set_strategy_selection_method(training_strategy_selector)
    self._meta_strategy_str = meta_strategy_method
    self._nash_solver_path = kwargs['nash_solver_path'] or None
    self._symmetric = symmetric
    self._quiesce = kwargs['quiesce'] or False
    super(GenEGTASolver, self).__init__(game, oracle, initial_policies,
                                        meta_strategy_method, **kwargs)
    

  def _initialize_policy(self, initial_policies):
    self._policies = [[] for k in range(self._num_players)]

    # Creating a tabular policy for an N player games is very costly. We only
    # create one, and copy its attributes to the other policies.
    # Each player starts with a uniform random policy.

    self._new_policies = [
        ([initial_policies[k]] if initial_policies else
         [UniformAgent(self._game)])
        for k in range(self._num_players)
    ]
    self._complete_ind = [[] for _ in range(self._num_players)]

  def _initialize_game_state(self):
    self._meta_games = [
        np.array(empty_list_generator(self._num_players))
        for _ in range(self._num_players)
    ]

    self.update_empirical_gamestate(seed=None,add_sample=True)

  def load(iter,checkpoint_dir,attr_file=None):
    """
    Load previous training results
    Params  :
    iter          : number of iter to start from
    checkpoint_dir: directory for tf graphs to be stored
    attr_file     : stores self._meta_games
    """
    self._policies = self._oracle.load(it,checkpoint_dir)
    if attr_file is not None:
      pklfile = open(attr_file,'rb')
      data = pickle.load(pklfile)
      self._meta_games = data[0]
      pklfile.close()
    else:
      self._fill_meta_games()
     
  def _fill_meta_games(self):
    # fill in meta game matrix, nash conv matrix is automatically filled
    # Used after loading the games
    num_of_policy = len(self._policies[0])
    self._meta_games = [
        np.ones([num_of_policy for _ in range(self._num_players)])*np.infty for _ in range(self._num_players)
    ]
    iter_ind = itertools.product(list(range(num_of_policy)),repeat=self._num_players)
    for ind in iter_ind:
        pol = [self._policies[i][ind[i]] for i in range(self._num_players)]
        self._meta_games[ind] = self.rl_sample_episodes(pol, self._sims_per_entry)
  
  def save_attr(self,iter,attr_file):
    """
    Save attributes: self._meta_games
    """
    pklfile = open(attr_file,'wb')
    pickle.dump([self._meta_games],pklfile)
    pklfile.close()

  def set_strategy_selection_method(self, training_strategy_selector):
    training_strategy_selector = training_strategy_selector or DEFAULT_STRATEGY_SELECTION_METHOD
    if isinstance(training_strategy_selector, str):
      self._training_strategy_selector = TRAINING_STRATEGY_SELECTORS.get(
          training_strategy_selector,
          TRAINING_STRATEGY_SELECTORS[DEFAULT_STRATEGY_SELECTION_METHOD])
    else:
      # We infer the argument passed is a callable function.
      self._training_strategy_selector = training_strategy_selector

  def update_meta_strategies(self):
      """Prevent double computation of nash probability
      """
      if self._meta_strategy_str=='nash' and hasattr(self,'_nash_prob'):
          self._meta_strategy_probabilities = self._nash_prob
      else:
          super(GenEGTASolver, self).update_meta_strategies(self._nash_solver_path)

  def update_rl_agents(self,iter):
      """Updates each agent using the RL oracle.
      Each player only adds one strategy at each PSRO iteration.
      Assume training_strategy_selector is 'probabilities'.
      """
      #if the game is symmetric, then players share the same strategy set.
      self._new_policies = []
      training_curves = []
      for current_player in range(self._num_players):
          current_new_policies = []
          new_policy,training_curve = self._oracle(
              iter,
              self._policies,
              current_player,
              self._meta_strategy_probabilities
              )
          current_new_policies.append(new_policy)
          training_curves.append(training_curve)
          if self._symmetric:
              for current_player in range(self._num_players):
                  self._new_policies.append(current_new_policies)
              break
          self._new_policies.append(current_new_policies)
      return {'train_iter'+str(iter)+'_p'+str(i):training_curves[i] for i in range(len(training_curves))}

  def inner_loop(self):
      found_confirmed_eq = False
      while not found_confirmed_eq:
          maximum_subgame = self.get_complete_meta_game
          ne_subgame = abstract_meta_trainer_egta.general_nash_strategy(self,self._nash_solver_path,maximum_subgame)
          # ne_support_num: list of list, index of where equilibrium is [[0,1],[2]]
          # cumsum: index ne_subgame with self._complete_ind
          cum_sum = [np.cumsum(ele) for ele in self._complete_ind]
          ne_support_num = []
          for i in range(self._num_players):
            ne_support_num_p = []
            for j in range(len(self._complete_ind[i])):
              if self._complete_ind[i][j]==1 and ne_subgame[i][cum_sum[i][j]-1]!=0:
                ne_support_num_p.append(j)
            ne_support_num.append(ne_support_num_p)
          # ne_subgame: non-zero equilibrium support, [[0.1,0.5,0.4],[0.2,0.4,0.4]]
          ne_subgame_nonzero = [np.array(ele) for ele in ne_subgame]
          ne_subgame_nonzero = [list(ele[ele!=0]) for ele in ne_subgame_nonzero]
          # get players' payoffs in nash equilibrium
          ne_payoffs = self.get_mixed_payoff(ne_support_num,ne_subgame_nonzero)
          # all possible deviation payoffs
          dev_pol, dev_payoffs = self.schedule_deviation(ne_support_num,ne_subgame_nonzero)
          # check max deviations and sample full subgame where beneficial deviation included
          dev = []
          maximum_subgame_index = [list(np.where(np.array(ele)==1)[0]) for ele in self._complete_ind]
          for i in range(self._num_players):
              if not len(dev_payoffs[i])==0 and max(dev_payoffs[i]) > ne_payoffs[i]:
                  pol = dev_pol[i][np.argmax(dev_payoffs[i])]
                  new_subgame_sample_ind = copy.deepcopy(maximum_subgame_index)
                  maximum_subgame_index[i].append(pol)
                  new_subgame_sample_ind[i] = [pol]
                  # add best deviation into subgame and sample it
                  for pol in itertools.product(*new_subgame_sample_ind):
                      self.sample_pure_policy_to_empirical_game(pol) 
                  dev.append(i)
                  # all other player's policies have to sample previous players' best deviation
          found_confirmed_eq = (len(dev)==0)
          # debug: check maximum subgame remains the same
          # debug: check maximum game reached

      # return confirmed nash equilibrium
      eq = []
      for p in range(self._num_players):
          eq_p = np.zeros([len(self._policies[p])],dtype=float)
          np.put(eq_p,ne_support_num[p],ne_subgame_nonzero[p])
          eq.append(eq_p)
      return eq 

  def schedule_deviation(self,eq,eq_sup):
      """
      Sample all possible deviation from eq
      Return a list of best deviation for each player
      if none for a player, return None for that player
      Params:
        eq     :  list of list, where equilibrium is.[[1],[0]]: an example of 2x2 game
        eq_sup :  Exact same shape with eq. Documents equilibrium support
      """
      devs = []
      dev_pol = []
      for p in range(self._num_players):
          # check all possible deviations
          dev = []
          possible_dev = list(np.where(np.array(self._complete_ind[p])==0)[0])
          iter_eq = copy.deepcopy(eq)
          iter_eq[p] = possible_dev
          for pol in itertools.product(*iter_eq):
              self.sample_pure_policy_to_empirical_game(pol)
          for pol in possible_dev:
              stra_li,stra_sup = copy.deepcopy(eq),copy.deepcopy(eq_sup)
              stra_li[p] = [pol]
              stra_sup[p] = [1.0]
              dev.append(self.get_mixed_payoff(stra_li,stra_sup)[p])
          devs.append(dev)
          dev_pol.append(possible_dev)
      return dev_pol, devs

  def get_mixed_payoff(self,strategy_list,strategy_support):
      """
      Check if the payoff exists. If not, return False
      Params:
        strategy_list    : [[],[]] 2 dimensional list, nash support policy index
        strategy_support : [[],[]] 2 dimensional list, nash support probability for corresponding strategies
      """
      if np.any(np.isnan(self._meta_games[0][np.ix_(*strategy_list)])):
        return False
      meta_game = [ele[np.ix_(*strategy_list)] for ele in self._meta_games]
      # multiple player prob_matrix tensor
      prob_matrix = np.outer(strategy_support[0],strategy_support[1])
      for i in range(self._num_players-2):
          ind = tuple([Ellipsis for _ in range(len(prob_matrix.shape))]+[None])
          prob_matrix = prob_matrix[ind]*strategy_support[i+2]
      payoffs=[]
      for i in range(self._num_players):
          payoffs.append(np.sum(meta_game[i]*prob_matrix))
      return payoffs
      
  def evaluate_nash_conv(self):
      """
      evaluate nash conv for the current nash equilibrium
      First compute NE of current empirical_game
      self._nash_prob    : [[],[]] 2 dimensional list
      ne_support         : [[],[]] 2 dimensional list, nash support policy index
      ind                : (,,) index of pure strategy profile, length num_of_player
      """
      nash_conv_mixed = 0
      if self._quiesce:
          self._nash_prob = self.inner_loop()
      else:
          self._nash_prob = abstract_meta_trainer_egta.general_nash_strategy(self,self._nash_solver_path)

      return 0
      
#      aggregator = policy_aggregator.PolicyAggregator(self._game)
#      aggr_policies = aggregator.aggregate(
#          range(self._num_players), self._policies, self._nash_prob)
#      exploitabilities = exploitability.nash_conv(
#          self._game, aggr_policies, return_only_nash_conv=True)
#      return exploitabilities

  def evaluate_iteration(self):
      """
      compute evaluation metrics of current iteration
      """
      metrics = {'nashconv':0}
      metrics['nashconv'] = self.evaluate_nash_conv()
      return metrics

  def iteration(self, 
                iter,
                seed=None
                ):
      """
      Override the iteration function in the AbstractMetaTrainer.
      :param seed: random seed.
      :param iter: current number of iteration
      """
      self._iterations += 1
      self.update_meta_strategies()  # Compute nash equilibrium.
      train_dict = self.update_rl_agents(iter)  # Generate new, Best Response agents via oracle.
      self.update_empirical_gamestate(seed=seed,add_sample=not self._quiesce)  # Update gamestate matrix.
      eval_dict = self.evaluate_iteration() # evaluate iteration using nashconv or combined game
      eval_dict.update(train_dict)
      return eval_dict

    #TODO: test this function.
  def rl_sample_episode(self, agents):
    """
    Samples an episode according to the policies, starting from state.
    :param agents: a list of policies.
    :return: a list of rewards, each per player.
    """
    env = rl_environment.Environment(self._game)

    # Uniform strategy has the same step function as rl agent.
    # No discount in evaluation mode.
    rewards = np.zeros(self._num_players, dtype=np.float32)
    with self._session.as_default():
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            if env.is_turn_based:
                agent_output = agents[player_id].step(time_step, is_evaluation=True)
                action_list = [agent_output.action]
            else:
                agents_output = [agent.step(time_step, is_evaluation=True) for agent in agents]
                action_list = [agent_output.action for agent_output in agents_output]
            time_step = env.step(action_list)
            #TODO: check the non-current player's reward not None.
            rewards += np.array(time_step.rewards, dtype=np.float32)

    return rewards

  def rl_sample_episodes(self, policies, num_episodes):
    """Samples episodes and averages their returns.

    Args:
      policies: A list of policies representing the policies executed by each
        player.
      num_episodes: Number of episodes to execute to estimate average return of
        policies.

    Returns:
      Average episode return over num episodes.
    """
    totals = np.zeros(self._num_players)
    for _ in range(num_episodes):
      totals += self.rl_sample_episode(policies).reshape(-1)
    return totals / num_episodes

  def update_complete_ind(self, policy_indicator, add_sample=True):
    """
    Find the maximum completed subgame with newly added policy(one policy for each player)
    Params:
    policy_indicator: policy to check, one number for each player
    add_sample      : whether there are sample data added after last update
    """
    for i in range(self._num_players):
      for _ in range(len(self._policies[i])-len(self._complete_ind[i])):
        self._complete_ind[i].append(0)

      if not add_sample or self._complete_ind[i][policy_indicator[i]]==1:
        continue
      selector = [list(np.where(np.array(ele)==1)[0]) for ele in self._complete_ind]
      selector[i].append(policy_indicator[i])
      if not np.any(np.isnan(self._meta_games[i][np.ix_(*selector)])):
        self._complete_ind[i][policy_indicator[i]]=1

  def sample_pure_policy_to_empirical_game(self, policy_indicator):
      """
      add data samples to data grid
      Params:
        policy_indicator: 1 dim list, containing poicy to sample for each player
      """
      if not np.isnan(self._meta_games[0][tuple(policy_indicator)]):
        return False
      estimated_policies = [self._policies[i][policy_indicator[i]] for i in range(self._num_players)]
      utility_estimates = self.rl_sample_episodes(estimated_policies,self._sims_per_entry)
      for k in range(self._num_players):
        self._meta_games[k][tuple(policy_indicator)] = utility_estimates[k]
      self.update_complete_ind(policy_indicator,add_sample=True)    
      return True

  def update_empirical_gamestate(self, seed=None, add_sample=False):
    """Given new agents in _new_policies, update meta_games through simulations.

    Args:
      seed: Seed for environment generation.

    Returns:
      Meta game payoff matrix.
    """
    if seed is not None:
      np.random.seed(seed=seed)
    assert self._oracle is not None

    # Concatenate both lists.
    updated_policies = [
        self._policies[k] + self._new_policies[k]
        for k in range(self._num_players)
    ]

    # Each metagame will be (num_strategies)^self._num_players.
    # There are self._num_player metagames, one per player.
    total_number_policies = [
        len(updated_policies[k]) for k in range(self._num_players)
    ]
    number_older_policies = [
        len(self._policies[k]) for k in range(self._num_players)
    ]
    number_new_policies = [
        len(self._new_policies[k]) for k in range(self._num_players)
    ]

    # Initializing the matrix with nans to recognize unestimated states.
    meta_games = [
        np.full(tuple(total_number_policies), np.nan)
        for k in range(self._num_players)
    ]

    # Filling the matrix with already-known values.
    older_policies_slice = tuple(
        [slice(len(self._policies[k])) for k in range(self._num_players)])
    for k in range(self._num_players):
      meta_games[k][older_policies_slice] = self._meta_games[k]

    if add_sample:
        # Filling the matrix for newly added policies.
        for current_player in range(self._num_players):
          # Only iterate over new policies for current player ; compute on every
          # policy for the other players.
          range_iterators = [
              range(total_number_policies[k]) for k in range(current_player)
          ] + [range(number_new_policies[current_player])] + [
              range(total_number_policies[k])
              for k in range(current_player + 1, self._num_players)
          ]
          for current_index in itertools.product(*range_iterators):
            used_index = list(current_index)
            used_index[current_player] += number_older_policies[current_player]
            if np.isnan(meta_games[current_player][tuple(used_index)]):
              estimated_policies = [
                  updated_policies[k][current_index[k]]
                  for k in range(current_player)
              ] + [
                  self._new_policies[current_player][current_index[current_player]]
              ] + [
                  updated_policies[k][current_index[k]]
                  for k in range(current_player + 1, self._num_players)
              ]
              # change to the rl_sample_episode.
              utility_estimates = self.rl_sample_episodes(estimated_policies, self._sims_per_entry)
              for k in range(self._num_players):
                meta_games[k][tuple(used_index)] = utility_estimates[k]

    assert sum(number_new_policies)==len(number_new_policies)

    self._meta_games = meta_games
    self._policies = updated_policies

    self.update_complete_ind(number_older_policies,add_sample=add_sample)

    return meta_games

  @property
  def get_meta_game(self):
    """Returns the meta game matrix."""
    return self._meta_games

  @property
  def get_complete_meta_game(self):
    """
    Returns the maximum complete game matrix
    in the same form as empirical game
    """
    selector = []
    for i in range(self._num_players):
      selector.append(list(np.where(np.array(self._complete_ind[i])==1)[0]))
    complete_subgame = [self._meta_games[i][np.ix_(*selector)] for i in range(self._num_players)] 
    return complete_subgame

  @property
  def get_policies(self):
    """Returns the players' policies."""
    return self._policies

  @property
  def get_strategy_computation_and_selection_kwargs(self):
    return self._strategy_computation_and_selection_kwargs

