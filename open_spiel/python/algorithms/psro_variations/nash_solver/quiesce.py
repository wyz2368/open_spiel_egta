from gameanalysis import collect

def empty_list_generator(number_dimensions):
    result = []
    for _ in range(number_dimensions-1):
        result = [result]
    return result

class EmpiricalGameUtility(object):
    """
    Empirical game noisy payoff utility table
    Implements incomplete nash equilibrium search with ECVI
    Refer to "Choosing Samples to Compute Heuristic-Strategy Nash Equilibrium" (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6599&rep=rep1&type=pdf)
    """
    def __init__(self,num_players,regret_thresh,gamma,gamma_star):
        """
        gamma     : numb of required samples for deviation
        gamma_star: numb of required samples for confirmed eq
        """
        self._regret_thresh = regret_thresh
        self._gamma = gamma
        self._gamma_star = gamma_star
        self._num_players = num_players
        self._mean = [
            np.array(empty_list_generator(num_players),dtype=float) for _ in range(num_players)]
        self._var = self._mean
        self._count = np.array(empty_list_generator(num_players),dtype=int)
        self._sample_buffer = [np.array(empty_list_generator(num_players+1)) for _ in range(num_players)]

    def add_strategy_grid(self,role_strats=[],samples=0):
        """
        Params:
            strats  :list of int, num of strategies for each player
            samples :numer of samples to add
        """
        assert len(role_strats)==len(self._mean),'strategy list should be of the same length with players'
        for p in range(self.num_players):
            self._count = np.append(slef._count,
                np.ones(,dtype=int),axis=p)
        self._mean = 
        self._count = 
        if samples != 0:
          self.add_samples(rest,samples)
    
    def add_samples(self,rests,,sims_per_entry=1):
        for rest in rests:
            policy = translate_policy(rest)
            utility_estimates = 

    def update_empirical_gamestate(self,mean,var,count):
        self._mean = mean
        self. 
    
    def update_ecvi(self):
        eq_in_max_subgame
    
    @property
    def get_meta_game(self):
        return self._mean

    @property
    def get_meta_game_var(self):
        return self._var

    @property
    def get_meta_game_count(self):
        return self._count

def incomplete_info_nash_search(eg,initial_restrictions=None,regret_thresh=1e-3,dist_thresh=0.1,support_thresh=1e-4,restricted_game_size=3,num_equilibria=1,num_backups=1,devs_by_role=False):
    equilibria = []
    required_subgame = []
    exp_restrictions = []
    backups = [[] for _ in range()]

    while len(confirmed_eq)==0 or 
    return nash
