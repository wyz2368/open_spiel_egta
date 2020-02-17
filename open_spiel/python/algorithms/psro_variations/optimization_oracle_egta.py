from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyspiel
import numpy as np
# from open_spiel.python.algorithms.psro_variations import abstract_meta_trainer
from open_spiel.python.algorithms.psro_variations.optimization_oracle import AbstractOracle
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import policy_gradient
# from open_spiel.python.algorithms.losses import rl_losses
# import tensorflow.compat.v1 as tf

#TODO: A global session is needed.
class RLoracle(AbstractOracle):
    """Oracle using RL to compute BR to policies."""
    def __init__(self,
                 session,
                 number_episodes_sampled=100,
                 **kwargs):
        super(RLoracle, self).__init__(**kwargs)

        self._number_episodes_sampled = number_episodes_sampled
        self._session = session

    def probabilistic_strategy_selector(self, total_policies, probabilities_of_playing_policies):
      """Returns [kwargs] policies randomly, proportionally with selection probas.
      """
      # By default, select only 1 new policy to optimize from.
      num_players = len(total_policies)
      meta_strategy_probabilities = probabilities_of_playing_policies
      used_policies = []
      for k in range(num_players):
        current_policies = total_policies[k]
        current_selection_probabilities = meta_strategy_probabilities[k]
        selected_policies = list(
          np.random.choice(
            current_policies,
            replace=False,
            p=current_selection_probabilities))
        used_policies.append(selected_policies)
      return used_policies


    def __call__(self,
                 game,
                 total_policies,
                 current_player,
                 probabilities_of_playing_policies,
                 **oracle_specific_execution_kwargs):
        '''
        Return a RL best response against a set of policies.
        :param game:
        :param total_policies:
        :param current_player:
        :param probabilities_of_playing_policies:
        :param oracle_specific_execution_kwargs:
        :return:
        '''
        assert type(game) == pyspiel.Game

        # num_players = game.num_players()
        env = rl_environment.Environment(game)

        info_state_size = env.observation_spec()["info_state"][0]
        num_actions = env.action_spec()["num_actions"]

        with self._session.as_default():
          training_agent = policy_gradient.PolicyGradient(  # pylint: disable=g-complex-comprehension
                self._session,
                player_id=current_player,
                info_state_size=info_state_size,
                num_actions=num_actions,
                loss_str="a2c",
                hidden_layers_sizes=[8, 8],
                batch_size=16,
                entropy_cost=0.001,
                critic_learning_rate=0.01,
                pi_learning_rate=0.01,
                num_critic_before_pi=4)

          #TODO: check all the players' id correct.
          for _ in range(self._number_episodes_sampled):

            agents = self.probabilistic_strategy_selector(total_policies, probabilities_of_playing_policies)

            time_step = env.reset()
            while not time_step.last():
              player_id = time_step.observations["current_player"]
              if env.is_turn_based:
                ## Training the current player's agent using step which updates the networks.
                if current_player == player_id:
                  agent_output = training_agent.step(time_step)
                else:
                  agent_output = agents[player_id].step(time_step, is_evaluation=True)
                action_list = [agent_output.action]
              else:
                agents_output = []
                for pid, agent in enumerate(agents):
                  if pid == current_player:
                    agents_output.append(training_agent.step(time_step))
                  else:
                    agents_output.append(agent.step(time_step, is_evaluation=True))

                action_list = [agent_output.action for agent_output in agents_output]
              time_step = env.step(action_list)

            # Episode is over, step the agent under training with final info state.
              training_agent.step(time_step)

        return training_agent






























