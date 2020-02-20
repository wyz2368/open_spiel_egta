from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
# import warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# warnings.filterwarnings("ignore")
import os
import datetime
from absl import app
from absl import flags
from absl import logging

from open_spiel.python.algorithms.psro_variations import optimization_oracle_egta
from open_spiel.python.algorithms.psro_variations import generalized_egta
import pyspiel
import tensorflow.compat.v1 as tf

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS
flags.DEFINE_string("game", "kuhn_poker", "Name of the game.")
flags.DEFINE_integer("sims_per_entry", 5,
                     "Number of simulations to update meta game matrix.")
flags.DEFINE_integer("psro_iterations", 2, "Number of iterations.")
flags.DEFINE_integer(
    "number_episodes_sampled", int(1e6),
    "Number of episodes per policy sampled for value estimation.")
flags.DEFINE_string("oracle","a2c","type of rl algorithm to use for oracle")
flags.DEFINE_string("root_result_folder",'root_result',"root directory of saved results")

def main(unused_argv):

  if not os.path.exists(FLAGS.root_result_folder):
    os.makedirs(FLAGS.root_result_folder)

  checkpoint_dir = os.path.join(os.getcwd(),FLAGS.root_result_folder,FLAGS.game+'_'+FLAGS.oracle+'_sims_'+str(FLAGS.sims_per_entry)+'_it'+str(FLAGS.psro_iterations)+'_ep'+str(FLAGS.number_episodes_sampled)+'_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
  os.makedirs(checkpoint_dir)

  game = pyspiel.load_game(FLAGS.game)

  global_sess = tf.Session()

  oracle = optimization_oracle_egta.RLoracle(
      session=global_sess,
      number_episodes_sampled=FLAGS.number_episodes_sampled,
      checkpoint_dir=checkpoint_dir)

  rl_solver = generalized_egta.GenEGTASolver(
      game=game,
      oracle=oracle,
      session=global_sess,
      sims_per_entry=FLAGS.sims_per_entry)
  
  #import warnings
  #warnings.simplefilter('error')

  for iter in range(FLAGS.psro_iterations):
    rl_solver.iteration(iter=iter)
    nash_probabilities = rl_solver.get_and_update_meta_strategies()
    print("{} / {}".format(iter + 1, FLAGS.psro_iterations))
    for ele in nash_probabilities:
      print(ele)

  print(FLAGS.game + " Meta Game Values")
  print(rl_solver.get_meta_game)
  print("")


if __name__ == "__main__":
  app.run(main)
