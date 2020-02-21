from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyspiel
import numpy as np
from tqdm import tqdm
import sys
# from open_spiel.python.algorithms.psro_variations import abstract_meta_trainer
import tensorflow.compat.v1 as tf
from open_spiel.python.algorithms.psro_variations.optimization_oracle import AbstractOracle
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import policy_gradient
from open_spiel.python.algorithms import dqn
# from open_spiel.python.algorithms.losses import rl_losses

class RLoracle(AbstractOracle):
    """Oracle using RL to compute BR to policies."""
    def __init__(self,
                 session,
                 oracle='a2c',
                 number_episodes_sampled=100,
		             checkpoint_dir=None,
                 **kwargs):
        super(RLoracle, self).__init__(**kwargs)

        self._number_episodes_sampled = number_episodes_sampled
        self._session = session
        self._oracle = oracle
        self._checkpoint_dir = checkpoint_dir

    def probabilistic_strategy_selector(self, total_policies, probabilities_of_playing_policies):
      """
      Returns [kwargs] policies randomly, proportionally with selection probas.
      """
      # By default, select only 1 new policy to optimize from.
      
      num_players = len(total_policies)
      meta_strategy_probabilities = probabilities_of_playing_policies
      used_policies = []
      for k in range(num_players):
        current_policies = total_policies[k]
        current_selection_probabilities = meta_strategy_probabilities[k]
        selected_policies = np.random.choice(
            current_policies,
            replace=False,
            p=current_selection_probabilities)
        used_policies.append(selected_policies)
      return used_policies


    def __call__(self,
                 game,
                 iter,
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

        num_players = game.num_players()
        env = rl_environment.Environment(game)

        info_state_size = env.observation_spec()["info_state"][0]
        num_actions = env.action_spec()["num_actions"]
        scope_name = 'iter'+str(iter)+'_'+'p'+str(current_player)
        with self._session.as_default():
          with tf.variable_scope(scope_name) as scope:
            if self._oracle == "dqn":
                  training_agent = dqn.DQN(
                        session=self._session,
                        player_id=current_player,
                        state_representation_size=info_state_size,
                        num_actions=num_actions,
                        hidden_layers_sizes=[64, 64],
                        replay_buffer_capacity=1e5,
                        batch_size=32)
            elif self._oracle in ["rpg", "qpg", "rm", "a2c"]:
                training_agent = policy_gradient.PolicyGradient(  # pylint: disable=g-complex-comprehension
                      session=self._session,
                      player_id=current_player,
                      info_state_size=info_state_size,
                      num_actions=num_actions,
                      loss_str=self._oracle,
                      hidden_layers_sizes=[128, 128],
                      batch_size=32,
                      entropy_cost=0.001,
                      critic_learning_rate=0.01,
                      pi_learning_rate=0.01,
                      num_critic_before_pi=32)
            else:
                raise ValueError("Oracle selected is not supported.")
          # saver and initize for all variables within scope
          saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope_name))
          for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name):
            var.initializer.run()

          episodes_rewards = []
          avg_rewards = []
          #TODO: check all the players' id correct.
          for ep in tqdm(range(self._number_episodes_sampled), file=sys.stdout):

            agents = self.probabilistic_strategy_selector(total_policies, probabilities_of_playing_policies)
            episode_reward = 0

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
              episode_reward += time_step.rewards[current_player]
              
            # Episode is over, step the agent under training with final info state, log episode reward
            training_agent.step(time_step)

            episodes_rewards.append(episode_reward)
            avg_rewards.append(sum(episodes_rewards[ep-100:ep])/100 if ep>99 else sum(episodes_rewards)/len(episodes_rewards))

          if self._checkpoint_dir:
            saver.save(self._session,self._checkpoint_dir+'/'+scope_name)

        return training_agent,avg_rewards
