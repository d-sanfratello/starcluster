import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mpl_toolkits import mplot3d
from collections import Counter

from scipy.stats import invwishart
from scipy.stats import t as student_t

def cluster_properties(stars, cluster_assignments, priors):
    
    L, k, nu, mu = priors
    
    if nu < 6:
        print('Warning: nu must be >= 6.')
        nu = 6

    cluster_labels = set(cluster_assignments)
    cluster_labels.discard(-1)
    
    cluster_pars = {}
    cluster_pdf  = {}
    
    for lab in cluster_assignments:
    
        stars_in_cluster = np.array(stars[np.where(cluster_labels == lab)])

        M    = stars_in_cluster.mean()
        S    = stars_in_cluster.cov()
        N    = len(stars_in_cluster)
        
        k_n  = k + N
        mu_n = np.atleast_2d((mu*k + N*M)/k_n)
        nu_n = nu + N
        L_n  = L + S*N + k*N*np.matmul((M - mu).T, (M - mu))/k_n
        
        t_df    = nu_n - 6 + 1
        t_shape = L_n*(k_n+1)/(k_n*t_df)

        s = invwishart(df = nu_n, scale = L_n).pdf
        m = student_t(df = t_df, loc = mu_n.flatten(), shape = t_shape).pdf
        
        cluster_pars[lab] = {'mu': mu_n, 'k': k_n, 'nu': nu_n, 'L': L_n}
        cluster_pdf[lab]  = {'mean': m, 'variance': s}
    
    return cluster_pars, cluster_pdf

def background_probability(assignments):

    N_draws = assignments.shape[1]
    counts  = np.count_nonzero(assignments == -1, axis = 1)
    p_bkg   = (counts + 0.5)/(N_draws + 1)
    return p_bkg, 1-p_bkg

def most_probable_cluster(assignments):

    clusters = np.zeros(len(assignments))
    for i, star in enumerate(assignments):
        clusters[i] = max(star, key=Counter(star).get)
        
    cluster_labels = set(clusters)
    cluster_labels.discard(-1)
    for i, lab in enumerate(cluster_labels):
        clusters[np.where(clusters == lab)] = i

    return clusters

def plot_clusters(stars, assignments, output_folder):
    
    p_bkg, p_cluster = background_probability(assignments)
    clusters         = most_probable_cluster(assignments)
    cluster_labels   = set(clusters)
    cluster_labels.discard(-1)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    c = ax.scatter(stars[:,0], stars[:,1], stars[:,2], c = p_bkg, cmap = 'coolwarm', marker = '.', s = 3)
    fig.colorbar(c, label = '$p_{bkg}$')
    fig.savefig(Path(output_folder, 'p_bkg.pdf'), bbox_inches = 'tight')

#    fig = plt.figure()
#    ax = fig.add_subplot(projection = '3d')
#    cmap = plt.get_cmap('gist_rainbow', len(cluster_labels))
#    for i, lab in enumerate(cluster_labels):
#        #FIXME: check c = lab
#        c = ax.scatter(stars[np.where(clusters == lab),0], stars[np.where(clusters == lab),1], stars[np.where(clusters == lab),2], c = lab, cmap = cmap, marker = '.', s = 3)
#    fig.colorbar(c, label = 'Cluster labels')
#    fig.savefig(Path(output_folder, 'clusters.pdf'), bbox_inches = 'tight')
    
    return
