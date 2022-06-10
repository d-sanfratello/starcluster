import numpy as np
import optparse as op
from scipy.stats import multivariate_normal as mn
from pathlib import Path
from tqdm import tqdm

def read_data():
    pass
    
def main():
    
    parser = op.OptionParser()
    parser.add_option("-i", "--input", type = "string", dest = "input_folder", help = "Folder/file with means and covariances")
    parser.add_option("-o", "--output", type = "string", dest = "output_folder", help = "Output folder", default = '.')
    parser.add_option("--n_samples", type = "int", dest = "n_samples", help = "Number of samples for each star", default = 1000)

    (options, args) = parser.parse_args()
    
    options.input_folder = Path(options.input_folder).resolve()
    options.output_folder = Path(options.output_folder).resolve()
    if not options.output_folder.exists():
        options.output_folder.mkdir(parents=True)
    
    means, covs = read_data(options.input_folder)
    
    for i, (m, s) in tqdm(enumerate(zip(means, covs)), total = len(means), desc = 'Stars'):
        samples = mn(m, s).rvs(options.n_samples)
        np.savetxt(Path(options.output_folder, '{0}.txt'.format(i)), samples)
