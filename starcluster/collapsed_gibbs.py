import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from collections import namedtuple, Counter

from scipy.special import gammaln

from starcluster.postprocess import plot_clusters

from numba import jit, njit
from numba.extending import get_cython_function_address
import ctypes

"""
Implemented as in https://dp.tdhopper.com/collapsed-gibbs/
"""

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(_dble, _dble)
gammaln_float64 = functype(addr)

@njit
def numba_gammaln(x):
  return gammaln_float64(x)
  
@jit
def my_student_t(df, t, mu, sigma, dim, s2max = np.inf):
    """
    http://gregorygundersen.com/blog/2020/01/20/multivariate-t/
    """
    vals, vecs = np.linalg.eigh(sigma)
    vals       = np.minimum(vals, s2max)
    logdet     = np.log(vals).sum()
    valsinv    = np.array([1./v for v in vals])
    U          = vecs * np.sqrt(valsinv)
    dev        = t - mu
    maha       = np.square(np.dot(dev, U)).sum(axis=-1)

    x = 0.5 * (df + dim)
    A = numba_gammaln(x)
    B = numba_gammaln(0.5 * df)
    C = dim/2. * np.log(df * np.pi)
    D = 0.5 * logdet
    E = -x * np.log(1 + (1./df) * maha)
    
    return A - B - C - D + E

class StarClusters:

    def __init__(self, burnin,
                       n_draws,
                       n_steps,
                       alpha0 = 1,
                       nu = 3,
                       k = 1/4.,
                       output_folder = './',
                       verbose = True,
                       initial_cluster_number = 5.,
                       sigma_max = None,
                       initial_assign = None,
                       rdstate = None,
                       ):

        if rdstate == None:
            self.rdstate = np.random.RandomState()
        else:
            self.rdstate = rdstate
            
        self.burnin  = burnin
        self.n_draws = n_draws
        self.n_steps = n_steps
        
        self.dim = 6
        
        if sigma_max is None:
            self.sigma_max_from_data = True
        else:
            self.sigma_max_from_data = False
            len_sigma_max = len(sigma_max)
            if len_sigma_max == 1:
                self.sigma_max = np.ones(self.dim)*sigma_max
            elif len_sigma_max == 2:
                self.sigma_max = np.concatenate((np.ones(self.dim//2)*sigma_max[0], np.ones(self.dim//2)*sigma_max[1]))
            elif len_sigma_max == 6:
                self.sigma_max = np.array(sigma_max)
            else:
                print('sigma_max has the wrong number of dimensions ({0}), allowed 1, 2 or 6.'.format(len_sigma_max))
                exit()
        # DP parameters
        self.alpha0 = alpha0
        # Student-t parameters
        self.nu  = np.max([nu, self.dim])
        self.k  = k
        # Miscellanea
        self.icn    = initial_cluster_number
        self.SuffStat = namedtuple('SuffStat', 'mean cov N')
        # Output
        self.output_folder = output_folder
        self.verbose = verbose
        
    def initial_state(self):
        '''
        Create initial state
        '''
        if self.initial_assign is not None:
            assign = self.initial_assign
        else:
            assign = np.array([int(a//(len(self.stars)/int(self.icn))) for a in range(len(self.stars))])
            # Randomly assign some stars to background
            assign[self.rdstate.choice(np.arange(len(assign)), size = len(assign)//(2*self.icn))] = 0
        
        cluster_ids = list(set(assign))
        state = {
            'cluster_ids_': cluster_ids,
            'data_': self.stars,
            'num_clusters_': len(cluster_ids),
            'alpha_': self.alpha0,
            'Ntot': len(self.stars),
            'hyperparameters_': {
                "L": self.L,
                "k": self.k,
                "nu": self.nu,
                "mu": self.mu
                },
            'suffstats': {cid: None for cid in cluster_ids},
            'assignment': assign,
            }
        self.state = state
        self.update_suffstats()
        return
    
    def update_suffstats(self):
        for cluster_id, N in Counter(self.state['assignment']).items():
            points_in_cluster = [x for x, cid in zip(self.state['data_'], self.state['assignment']) if cid == cluster_id]
            mean = np.atleast_2d(np.array(points_in_cluster).mean(axis = 0))
            cov  = np.cov(np.array(points_in_cluster), rowvar = False)
            N    = len(points_in_cluster)
            self.state['suffstats'][cluster_id] = self.SuffStat(mean, cov, N)
    
    def log_predictive_likelihood(self, data_id, cluster_id):
        '''
        Computes the probability of a sample to be drawn from a cluster conditioned on all the samples assigned to the cluster
        '''
        if cluster_id == "new":
            ss = self.SuffStat(np.atleast_2d(0),np.identity(self.dim)*0,0)
        else:
            ss  = self.state['suffstats'][cluster_id]
            
        x = self.state['data_'][data_id]
        mean = ss.mean
        S = ss.cov
        N = ss.N
        # Update hyperparameters
        k_n  = self.state['hyperparameters_']["k"] + N
        mu_n = (self.state['hyperparameters_']["mu"]*self.state['hyperparameters_']["k"] + N*mean)/k_n
        nu_n = self.state['hyperparameters_']["nu"] + N
        L_n  = self.state['hyperparameters_']["L"]*self.state['hyperparameters_']["k"] + S*N + self.state['hyperparameters_']["k"]*N*np.matmul((mean - self.state['hyperparameters_']["mu"]).T, (mean - self.state['hyperparameters_']["mu"]))/k_n
        # Update t-parameters
        t_df    = nu_n - self.dim + 1
        t_shape = L_n*(k_n+1)/(k_n*t_df)
        # Compute logLikelihood
        if cluster_id == 0:
            # FIXME: p(x,y,z) = cnst for background
            logL = my_student_t(df = t_df, t = np.atleast_2d(x[3:]), mu = mu_n[:,3:], sigma = t_shape[3:, 3:], dim = self.dim//2, s2max = np.inf) - self.log_volume
        else:
            logL = my_student_t(df = t_df, t = np.atleast_2d(x), mu = mu_n, sigma = t_shape, dim = self.dim, s2max = self.sigma_max)
        return logL[0]

    def add_datapoint_to_suffstats(self, x, ss):
        x = np.atleast_2d(x)
        mean = (ss.mean*(ss.N)+x)/(ss.N+1)
        cov  = (ss.N*(ss.cov + np.matmul(ss.mean.T, ss.mean)) + np.matmul(x.T, x))/(ss.N+1) - np.matmul(mean.T, mean)
        return self.SuffStat(mean, cov, ss.N+1)

    def remove_datapoint_from_suffstats(self, x, ss):
        x = np.atleast_2d(x)
        if ss.N == 1:
            return(self.SuffStat(np.atleast_2d(0),np.identity(self.dim)*0,0))
        mean = (ss.mean*(ss.N)-x)/(ss.N-1)
        cov  = (ss.N*(ss.cov + np.matmul(ss.mean.T, ss.mean)) - np.matmul(x.T, x))/(ss.N-1) - np.matmul(mean.T, mean)
        return self.SuffStat(mean, cov, ss.N-1)
        
    def cluster_assignment_distribution(self, data_id):
        """
        Compute the marginal distribution of cluster assignment
        for each cluster.
        """
        scores = {}
        cluster_ids = list(self.state['suffstats'].keys()) + ['new']
        for cid in cluster_ids:
            scores[cid] = self.log_predictive_likelihood(data_id, cid)
            scores[cid] += self.log_cluster_assign_score(cid)
        scores = {cid: np.exp(score) for cid, score in scores.items()}
        normalization = 1/sum(scores.values())
        scores = {cid: score*normalization for cid, score in scores.items()}
        return scores

    def log_cluster_assign_score(self, cluster_id):
        """
        Log-likelihood that a new point generated will
        be assigned to cluster_id given the current state.
        """
        if cluster_id == "new":
            return np.log(self.state["alpha_"])
        else:
            return np.log(self.state['suffstats'][cluster_id].N)

    def create_cluster(self):
        self.state["num_clusters_"] += 1
        cluster_id = max(self.state['suffstats'].keys()) + 1
        self.state['suffstats'][cluster_id] = self.SuffStat(np.atleast_2d(0),np.identity(self.dim)*0,0)
        self.state['cluster_ids_'].append(cluster_id)
        return cluster_id

    def destroy_cluster(self, cluster_id):
        self.state["num_clusters_"] -= 1
        del self.state['suffstats'][cluster_id]
        self.state['cluster_ids_'].remove(cluster_id)
        
    def prune_clusters(self):
        for cid in self.state['cluster_ids_']:
            if self.state['suffstats'][cid].N == 0:
                self.destroy_cluster(cid)

    def sample_assignment(self, data_id):
        """
        Sample new assignment from marginal distribution.
        If cluster is "new", create a new cluster.
        """
        scores = self.cluster_assignment_distribution(data_id).items()
        labels, scores = zip(*scores)
        cid = self.rdstate.choice(labels, p=scores)
        if cid == "new":
            return self.create_cluster()
        else:
            return int(cid)

    def update_alpha(self, thinning = 300):
        '''
        Update concentration parameter
        '''
        a_old = self.state['alpha_']
        n     = self.state['Ntot']
        K     = len(self.state['cluster_ids_'])
        for _ in range(thinning):
            a_new = a_old + self.rdstate.uniform(-1,1)*0.5
            if a_new > 0:
                logP_old = numba_gammaln(a_old) - numba_gammaln(a_old + n) + K * np.log(a_old) - 1./a_old
                logP_new = numba_gammaln(a_new) - numba_gammaln(a_new + n) + K * np.log(a_new) - 1./a_new
                if logP_new - logP_old > np.log(self.rdstate.uniform()):
                    a_old = a_new
        return a_old

    def gibbs_step(self):
        """
        Collapsed Gibbs sampler for Dirichlet Process Gaussian Mixture Model
        """
        self.state['alpha_'] = self.update_alpha()
        pairs = zip(self.state['data_'], self.state['assignment'])
        for data_id, (datapoint, cid) in enumerate(pairs):
            self.state['suffstats'][cid] = self.remove_datapoint_from_suffstats(datapoint, self.state['suffstats'][cid])
            self.prune_clusters()
            cid = self.sample_assignment(data_id)
            self.state['assignment'][data_id] = cid
            self.state['suffstats'][cid] = self.add_datapoint_to_suffstats(self.state['data_'][data_id], self.state['suffstats'][cid])

    def count_clusters(self):
        ids = np.array(self.state['assignment'])
        cnt = Counter(ids)
        return len(set([k for k in ids if cnt[k] > 1])) - 1
    
    def run_sampling(self):
        self.initial_state()
        for i in range(self.burnin):
            if self.verbose:
                print('\rBURN-IN: {0}/{1}'.format(i+1, self.burnin), end = '')
            self.gibbs_step()
        if self.verbose:
            print('\n', end = '')
        for i in range(self.n_draws):
            if self.verbose:
                print('\rSAMPLING: {0}/{1}'.format(i+1, self.n_draws), end = '')
            for _ in range(self.n_steps):
                self.gibbs_step()
            self.assignments.append(np.array(self.state['assignment']))
            self.n_clusters.append(self.count_clusters())
            self.alpha_samples.append(self.state['alpha_'])
        if self.verbose:
            print('\n', end = '')
        return
        
    def postprocess(self):
    
        assignments = np.array(self.assignments).T
        np.savetxt(Path(self.output_assignments, 'assignments.txt'), assignments)
        
        plot_clusters(self.stars, assignments, self.output_skymaps)
        
        fig, ax = plt.subplots()
        ax.plot(np.arange(1,len(self.n_clusters)+1), self.n_clusters, ls = '--', marker = ',', linewidth = 0.5)
        fig.savefig(Path(self.output_n_clusters, 'n_clusters_{0}.pdf'.format(self.e_ID)), bbox_inches='tight')
        
        fig, ax = plt.subplots()
        ax.hist(self.alpha_samples, bins = int(np.sqrt(len(self.alpha_samples))), histtype = 'step', density = True)
        fig.savefig(Path(self.output_alpha, 'alpha_{0}.pdf'.format(self.e_ID)), bbox_inches='tight')
    
    def make_folders(self):
    
        self.output_events = Path(self.output_folder, 'reconstructed_events')
        dirs       = ['n_clusters', 'assignments', 'alpha', 'skymaps']
        attr_names = ['output_n_clusters', 'output_assignments', 'output_alpha', 'output_skymaps']
        
        if not self.output_events.exists():
            self.output_events.mkdir()
        
        for d, attr in zip(dirs, attr_names):
            newfolder = Path(self.output_events, d)
            if not newfolder.exists():
                try:
                    newfolder.mkdir()
                except FileExistsError:
                    pass
            setattr(self, attr, newfolder)
        return
        

    def run(self, args):
        """
        Runs the sampler, saves samples and produces output plots.
        
        Arguments:
            :iterable args: Iterable with arguments. These are (in order):
                                - stars: the stars to analyse
                                - event id: name to be given to the data set
                                - log_volume: uniform prior on background stars
                                position
                                - inital assignment: initial guess for cluster assignments (optional - if it does not apply, use None)
        Returns:
            :tuple: mean and variance hyperpriors
        """
        
        # Unpack arguments
        stars = args[0]
        event_id = args[1]
        log_volume = args[2]
        initial_assign = args[3]
        
        self.stars          = stars
        self.initial_assign = initial_assign
        self.e_ID           = event_id
        # FIXME: p(x,y,z) = cnst for background
        self.log_volume     = log_volume

        if self.sigma_max_from_data:
            self.sigma_max = np.std(self.stars, axis = 0)/2.
        
        self.mu = np.atleast_2d(np.mean(self.stars, axis = 0))
        self.L  = self.k*(self.sigma_max/2.)**2*np.identity(self.dim)
        
        self.assignments = []
        self.alpha_samples = []
        self.mixture_samples = []
        self.n_clusters = []
        
        # Run the analysis
        self.make_folders()
        self.run_sampling()
        self.postprocess()
        
        return (self.L, self.k, self.nu, self.mu)
