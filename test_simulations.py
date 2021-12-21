import numpy as np
import ray
import optparse as op
from pathlib import Path
from ray.util import ActorPool

from starcluster.collapsed_gibbs import StarClusters

def read_catalog(file):
    data = np.genfromtxt(file, names = True)
    return np.array([data['x'], data['y'], data['z'], data['vx'], data['vy'], data['vz']]).T, data['cl_ID']

def compute_volume(catalog):
    return np.prod(np.max(catalog, axis = 0) - np.min(catalog, axis = 0))

def main():
    parser = op.OptionParser()
    
    # Input/Output
    parser.add_option("-i", "--input", type = "string", dest = "event_file", help = "Input file")
    parser.add_option("-o", "--output", type = "string", dest = "output", help = "Output folder")
    # Settings
    parser.add_option("--samp_settings", type = "string", dest = "samp_settings", help = "Burnin, number of draws and number of steps between draws", default = '100,100,10')
    parser.add_option("--icn", dest = "initial_cluster_number", type = "float", help = "Initial cluster number", default = 5.)
    parser.add_option("-s", "--seed", dest = "seed", type = "float", default = 0, help = "Fix seed for reproducibility")
    # Priors
    parser.add_option("--prior", type = "string", dest = "prior", help = "Parameters for NIG prior (nu0, k0). See https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf sec. 6 for reference", default = '1,1')
    parser.add_option("--alpha", type = "float", dest = "alpha0", help = "Internal (event) initial concentration parameter", default = 1.)
    parser.add_option("--sigma_max_pos", type = "float", dest = "sigma_max_pos", help = "Maximum std for clusters", default = 5)
    parser.add_option("--sigma_max_vel", type = "float", dest = "sigma_max_vel", help = "Maximum std for clusters", default = 5)
    
    (options, args) = parser.parse_args()
    
    # Converts relative paths to absolute paths
    options.event_file = Path(str(options.event_file)).absolute()
    options.output     = Path(str(options.output)).absolute()
    
    # Read catalog and compute volume
    catalog, truths = read_catalog(options.event_file)
    volume          = compute_volume(catalog)
    
    # Create a RandomState
    if options.seed == 0:
        rdstate = np.random.RandomState(seed = 1)
    else:
        rdstate = np.random.RandomState()
    
    if options.prior is not None:
        options.nu, options.k = [float(x) for x in options.prior.split(',')]
    options.burnin, options.n_draws, options.n_steps = (int(x) for x in options.samp_settings.split(','))
    
    try:
        ray.init(num_cpus = 1)
    except:
        ray.init(num_cpus = 1, object_store_memory=10**9)
    
    sampler = StarClusters.remote(
                            burnin  = options.burnin,
                            n_draws = options.n_draws,
                            n_steps = options.n_steps,
                            alpha0  = float(options.alpha0),
                            nu      = options.nu,
                            k       = options.k,
                            output_folder = str(options.output),
                            verbose = 1,
                            initial_cluster_number = int(options.initial_cluster_number),
                            sigma_max = np.array([float(options.sigma_max_pos), float(options.sigma_max_vel)]),
                            rdstate = rdstate,
                            )
    pool = ActorPool([sampler])
    priors = []
    for s in pool.map(lambda a, v: a.run.remote(v), [[catalog, 'mock_data', volume, None]]):
        priors.append(s)
    


if __name__ == '__main__':
    main()
    
