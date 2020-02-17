import numpy as np
from open_spiel.python.algorithms.psro_variations.nash_solver import subproc
import os
import pickle
import itertools
import logging


#TODO: adapt to general-sum many-player game.
#TODO: change path.join

def isExist(path):
    return os.path.exists(path)

def save_pkl(obj,path):
    with open(path,'wb') as f:
        pickle.dump(obj,f)

def load_pkl(path):
    if not isExist(path):
        raise ValueError(path + " does not exist.")
    with open(path,'rb') as f:
        result = pickle.load(f)
    return result

gambit_DIR = os.getcwd() + '/nfg/payoffmatrix.nfg'

def encode_gambit_file(meta_games):
    num_players = len(meta_games)
    # Write header
    with open(gambit_DIR, "w") as nfgFile:
        nfgFile.write('NFG 1 R "Empirical Game"\n')
        name_players = '{ "p1"'
        for i in range(2,num_players+1):
            name_players += " " + "\"" + 'p' + str(i) + "\""
        name_players += ' }'
        nfgFile.write(name_players)
        # Write strategies
        num_strs = '{ '
        range_iterators = []
        for i in np.shape(meta_games[0]):
            num_strs += str(i) + " "
            range_iterators += [range(i)]
        num_strs += '}'
        nfgFile.write(num_strs + '\n\n')
        # Write outcomes
        for current_index in itertools.product(*range_iterators):
            for meta_game in meta_games:
                nfgFile.write(str(meta_game[tuple(current_index)]) + " ")

def gambit_analysis(timeout):
    if not isExist(gambit_DIR):
        raise ValueError(".nfg file does not exist!")
    command_str = "gambit-gnm -q " + os.getcwd() + "/nfg/payoffmatrix.nfg -d 8 > " + os.getcwd() + "/nfg/nash.txt"
    subproc.call_and_wait_with_timeout(command_str, timeout)

def gambit_analysis_pure(timeout):
    if not isExist(gambit_DIR):
        raise ValueError(".nfg file does not exist!")
    command_str = "gambit-enumpure -q " + os.getcwd() + "/nfg/payoffmatrix.nfg > " + os.getcwd() + "/nfg/nash.txt"
    subproc.call_and_wait_with_timeout(command_str, timeout)

def decode_gambit_file(meta_games, mode="all", max_num_nash=10):
    nash_DIR = os.getcwd() + '/nfg/nash.txt'
    if not isExist(nash_DIR):
        raise ValueError("nash.txt file does not exist!")
    num_lines = file_len(nash_DIR)

    logging.info("Number of NE is ", num_lines)
    if max_num_nash != None:
        if num_lines >= max_num_nash:
            num_lines = max_num_nash
            logging.info("Number of NE is constrained by the num_nash.")

    shape = np.shape(meta_games[0])
    slice_idx = []
    pos = 0
    for i in range(len(shape)):
        slice_idx.append(range(pos, pos + shape[i]))
        pos += shape[i]

    equilibria = []
    with open(nash_DIR,'r') as f:
        for _ in np.arange(num_lines):
            equilibrim = []
            nash = f.readline()
            if len(nash.strip()) == 0:
                continue
            nash = nash[3:]
            nash = nash.split(',')
            new_nash = []
            for j in range(len(nash)):
                new_nash.append(convert(nash[j]))

            new_nash = np.array(new_nash)
            new_nash = np.round(new_nash, decimals=8)
            for idx in slice_idx:
                equilibrim.append(new_nash[idx])
            equilibria.append(equilibrim)

    if mode == "all" or mode == "pure":
        return equilibria
    elif mode == "one":
        return equilibria[0]
    else:
        logging.info("mode is beyond all/pure/one.")


def do_gambit_analysis(meta_games, mode, timeout = 600):
    encode_gambit_file(meta_games)
    while True:
        if mode == 'pure':
            gambit_analysis_pure(timeout)
        else:
            gambit_analysis(timeout)
        # If there is no pure NE, find mixed NE.
        nash_DIR = os.getcwd() + '/nfg/nash.txt'
        if not isExist(nash_DIR):
            raise ValueError("nash.txt file does not exist!")
        num_lines = file_len(nash_DIR)
        if num_lines == 0:
            logging.info("Pure NE does not exist. Return mixed NE.")
            mode = 'all'
            continue
        equilibria = decode_gambit_file(meta_games, mode)
        if len(equilibria) != 0:
            break
        timeout += 120
        if timeout > 7200:
            logging.info("Gambit has been running for more than 2 hour.!")
        logging.info("Timeout has been added by 120s.")
    logging.info('gambit_analysis done!')
    return equilibria


def convert(s):
    try:
        return float(s)
    except ValueError:
        num, denom = s.split('/')
        return float(num) / float(denom)


# Get number of lines in a text file.
def file_len(fname):
    num_lines = sum(1 for line in open(fname))
    return num_lines



