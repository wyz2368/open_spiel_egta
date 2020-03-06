from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
# import warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# warnings.filterwarnings("ignore")

import os
import re
import datetime
from absl import app
import sys
from absl import flags
from absl import logging
import pyspiel
from tensorboardX import SummaryWriter

from open_spiel.python.algorithms.psro_variations import optimization_oracle_egta
from open_spiel.python.algorithms.psro_variations import generalized_egta
import tensorflow.compat.v1 as tf

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS
flags.DEFINE_string("game", "kuhn_poker", "Name of the game.")
flags.DEFINE_integer("sims_per_entry", 1,
                     "Number of simulations to update meta game matrix.")
flags.DEFINE_integer("psro_iterations", 150, "Number of iterations.")
flags.DEFINE_integer(
    "number_episodes_sampled", int(5e5),
    "Number of episodes per policy sampled for value estimation.")
flags.DEFINE_string("oracle","a2c","type of rl algorithm to use for oracle")
flags.DEFINE_string("root_result_folder",'root_result',"root directory of saved results")
flags.DEFINE_boolean("record_train",True,'record training curve of each player, each iteration')
flags.DEFINE_string("load_folder","","folder for load policy: the number of iteration will be determined")
flags.DEFINE_string("nash_solver_path","/home/qmaai/gambit_python3_supported/bin/","lrsnash executable filepath or bin folder for gambit")
#flags.DEFINE_string("nash_solver_path","","lrsnash executable filepath or bin folder for gambit")

def main(unused_argv):
  begin_iter = 0
  if FLAGS.load_folder!='':
      os.path.exists(FLAGS.load_folder)
      checkpoint_dir = FLAGS.load_folder
      # get last iteration number from filename
      for filename in os.listdir(checkpoint_dir):
          if 'iter' in filename:
              begin_iter = max(begin_iter,int(re.search(r'\d+',filename).group()))
  else:
      if not os.path.exists(FLAGS.root_result_folder):
          os.makedirs(FLAGS.root_result_folder)
      checkpoint_dir = os.path.join(os.getcwd(),FLAGS.root_result_folder,FLAGS.game+'_'+FLAGS.oracle+'_sims_'+str(FLAGS.sims_per_entry)+'_it'+str(FLAGS.psro_iterations)+'_ep'+str(FLAGS.number_episodes_sampled)+'_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
      os.makedirs(checkpoint_dir)
  #sys.stdout = open(checkpoint_dir+'/stdout.txt','a+')

  game = pyspiel.load_game(FLAGS.game)

  global_sess = tf.Session()

  oracle = optimization_oracle_egta.RLoracle(
      game = game,
      session=global_sess,
      number_episodes_sampled=FLAGS.number_episodes_sampled,
      checkpoint_dir=checkpoint_dir)

  rl_solver = generalized_egta.GenEGTASolver(
      game=game,
      oracle=oracle,
      session=global_sess,
      sims_per_entry=FLAGS.sims_per_entry,
      nash_solver_path=FLAGS.nash_solver_path)
  
  if FLAGS.load_folder!='':
    rl_solver.load(begin_iter,checkpoint_dir,checkpoint_dir+'/attr.pkl')
  
  #import warnings
  #warnings.simplefilter('error')
  writer = SummaryWriter(logdir=checkpoint_dir+'/log')
  for iter in range(begin_iter+1,FLAGS.psro_iterations):
    result_dict = rl_solver.iteration(iter=iter)
    rl_solver.save_attr(iter,checkpoint_dir+'/attr.pkl')
    nash_probabilities = rl_solver.get_and_update_meta_strategies()
    print("{} / {}".format(iter + 1, FLAGS.psro_iterations))
    for ele in nash_probabilities:
      print(ele)
     
    for key,val in result_dict.items():
      # train: each episode during an iter of episodes
      if 'train' in key:
        for i in range(len(result_dict[key])):
          writer.add_scalar(key,result_dict[key][i],i)
      elif 'nashconv' in key:
        writer.add_scalar(key,result_dict[key],iter)
      else: # not implemented yet
        pass
  writer.close()
  print(FLAGS.game + " Meta Game Values")
  print(rl_solver.get_meta_game)
  print("")


if __name__ == "__main__":
  app.run(main)
