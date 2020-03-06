from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pyspiel
import numpy as np
import joblib
from tqdm import tqdm
import sys
# from open_spiel.python.algorithms.psro_variations import abstract_meta_trainer
import tensorflow.compat.v1 as tf
from open_spiel.python.algorithms.psro_variations.optimization_oracle import AbstractOracle
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import policy_gradient
from open_spiel.python.algorithms import dqn
# from open_spiel.python.algorithms.losses import rl_losses


def save_variables(save_path, variables, sess):
    ps = sess.run(variables)
    save_dict = {v.name: value for v, value in zip(variables, ps)}
    dirname = os.path.dirname(save_path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)
    joblib.dump(save_dict, save_path)


def load_variables(load_path, variables, sess):
    loaded_params = joblib.load(os.path.expanduser(load_path))
    restores = []
    if isinstance(loaded_params, list):
        assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
        for d, v in zip(loaded_params, variables):
            restores.append(v.assign(d))
    else:
        for v in variables:
            restores.append(v.assign(loaded_params[v.name]))
    sess.run(restores)


class ARSoracle(AbstractOracle):
    """Oracle using ARS to compute BR to policies."""

    def __init__(self,
                 game,
                 session,
                 number_episodes_sampled=100,
                 checkpoint_dir=None,
                 **kwargs):
        super(ARSoracle, self).__init__(**kwargs)
        self._number_episodes_sampled = number_episodes_sampled
        self._session = session
        self._checkpoint_dir = checkpoint_dir

        assert type(game) == pyspiel.Game
        self._num_players = game.num_players()
        self._env = rl_environment.Environment(game)

    def probabilistic_strategy_selector(self, total_policies, probabilities_of_playing_policies):
        """
        Returns [kwargs] policies randomly, proportionally with selection probas.
        """
        # By default, select only 1 new policy to optimize from.

        meta_strategy_probabilities = probabilities_of_playing_policies
        used_policies = []
        for k in range(self._num_players):
            current_policies = total_policies[k]
            current_selection_probabilities = meta_strategy_probabilities[k]
            selected_policies = np.random.choice(
                current_policies,
                replace=False,
                p=current_selection_probabilities)
            used_policies.append(selected_policies)
        return used_policies

    def load(self, iter, checkpoint_dir):
        """
        Construct game graphs and then load them
        """
        policies = [[] for _ in range(self._num_players)]
        with self._session.as_default():
            for it in range(iter):
                for current_player in range(self._num_players):
                    scope_name = 'iter' + str(iter) + '_' + 'p' + str(current_player)
                    graph = self.build_graph(scope_name,
                                             current_player,
                                             self._env.observation_spec()["info_state"][0],
                                             self._env.action_spec()["num_actions"])
                    load_variables(checkpoint_dir + '/' + scope_name,
                                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name),
                                   self._session)
                    policies[current_player].append(graph)
        return policies

    def build_ars(self,


        return training_agent

    def __call__(self,
                 iter,
                 total_policies,
                 current_player,
                 probabilities_of_playing_policies,
                 **oracle_specific_execution_kwargs):
        '''
        Return a RL best response against a set of policies.
        :param total_policies:
        :param current_player:
        :param probabilities_of_playing_policies:
        :param oracle_specific_execution_kwargs:
        :return:
        '''

        scope_name = 'iter' + str(iter) + '_' + 'p' + str(current_player)
        with self._session.as_default():
            training_agent = self.build_graph(scope_name,
                                              current_player,
                                              self._env.observation_spec()["info_state"][0],
                                              self._env.action_spec()["num_actions"])
            # saver and initize for all variables within scope
            for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name):
                var.initializer.run()

            episodes_rewards = []
            avg_rewards = []
            # TODO: check all the players' id correct.
            for ep in tqdm(range(self._number_episodes_sampled), file=sys.stdout):

                agents = self.probabilistic_strategy_selector(total_policies, probabilities_of_playing_policies)
                episode_reward = 0

                time_step = self._env.reset()
                while not time_step.last():
                    player_id = time_step.observations["current_player"]
                    if self._env.is_turn_based:
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
                    time_step = self._env.step(action_list)
                    episode_reward += time_step.rewards[current_player]

                # Episode is over, step the agent under training with final info state, log episode reward
                training_agent.step(time_step)

                episodes_rewards.append(episode_reward)
                avg_rewards.append(sum(episodes_rewards[ep - 100:ep]) / 100 if ep > 99 else sum(episodes_rewards) / len(
                    episodes_rewards))

            if self._checkpoint_dir:
                save_variables(self._checkpoint_dir + '/' + scope_name,
                               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name), self._session)

        return training_agent, avg_rewards