from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
# import warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# warnings.filterwarnings("ignore")

from absl import app
from absl import flags

from open_spiel.python.algorithms.psro_variations import optimization_oracle_egta
from open_spiel.python.algorithms.psro_variations import generalized_egta
import pyspiel
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("game", "kuhn_poker", "Name of the game.")
flags.DEFINE_integer("sims_per_entry", 2,
                     "Number of simulations to update meta game matrix.")
flags.DEFINE_integer("psro_iterations", 5, "Number of iterations.")
flags.DEFINE_integer(
    "number_episodes_sampled", 5,
    "Number of episodes per policy sampled for value estimation.")



def main(unused_argv):

  game = pyspiel.load_game(FLAGS.game)

  global_sess = tf.Session()

  oracle = optimization_oracle_egta.RLoracle(
      session=global_sess,
      number_episodes_sampled=FLAGS.number_episodes_sampled)

  rl_solver = generalized_egta.GenEGTASolver(
      game=game,
      oracle=oracle,
      session=global_sess,
      sims_per_entry=FLAGS.sims_per_entry)

  for iter in range(FLAGS.psro_iterations):
    rl_solver.iteration()
    nash_probabilities = rl_solver.get_and_update_meta_strategies()
    print("{} / {}".format(iter + 1, FLAGS.psro_iterations))
    print(nash_probabilities)

  meta_game = rl_solver.get_meta_game
  nash_probabilities = rl_solver.get_and_update_meta_strategies()

  print(FLAGS.game + " Nash probabilities")
  print(nash_probabilities)
  print("")

  print(FLAGS.game + " Meta Game Values")
  print(meta_game)
  print("")


if __name__ == "__main__":
  app.run(main)