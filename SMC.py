#Sequential Monte Carlo (SMC) algorithm for Bayesian inference.

import numpy as np
from scipy.optimize import bisect


### First we implement systematic resampling as defined in algorithm 2 of the paper. ###
def systematic_resampling(W):
    """
    Perform systematic resampling given a set of normalized weights.

    Args:
        W (numpy.ndarray): Array of normalized weights.
        W_t (float): Normalized weights of the current time step, size N.

    Returns:
       A (numpy.ndarray) : Indices of resampled particles.
    """
    N = len(W)
    U = np.random.uniform(0, 1)  # Step a: Sample U ~ U([0, 1])
    cumulative_weights = N * np.cumsum(W)  # Step b: Compute cumulative weights

    # Step c: Initialize variables
    s = U
    m = 1
    A = []
    # Step d: Resample
    for n in range(1, N + 1):
        while s > cumulative_weights[m]:
            m += 1
        A.append(m)
        s += 1 
    return np.array(A)


### Next we implement the SMC algorithm as defined in algorithm 1 of the paper. ###
def Tempering_SMC(N, tau, kappa, target_distribution, initial_distribution, likelihood_ratio):
    """ Perform the Tempering Sequential Monte Carlo (SMC) algorithm for Bayesian inference.

    Parameters
    ----------
    N : int
    Number of particles to use in the SMC algorithm.

    tau : float
        ESS threshold for resampling.

    rw : positive
        random walk tuning parameter.

    kappa : float
        Random walk tuning parameter.

    target_distribution : callable
        Target distribution function  pi

    initial_distribution : callable
        Initial distribution function pi_xi

    likelihood_ratio : callable
        Function computing r_n(theta)

    Returns:
    ----------
    list: List of particle states at each iteration. 
    """ 
    
    theta = [initial_distribution() for _ in range(N)]
    t = 1
    lambda_t = 0
    Z = 1
    results = []

    def program_bisec(lambd):
        w = np.exp(- (lambd - lambda_t) * np.array([likelihood_ratio(p) for p in theta]))
        ratio = (np.sum(w) ** 2) / np.sum(w ** 2)
        return ratio - N * tau
    
    while True:
        # Step a: Solve for new lambda_t using bisection 
        lambda_new = bisect(program_bisec, lambda_t, 1)
        w = np.exp(- (lambda_new - lambda_t) * np.array([likelihood_ratio(p) for p in theta]))
        if lambda_new <= lambda_t:
            break 

        # Step b: Resample
        normalized_weights = w / np.sum(w)  
        A = systematic_resampling(normalized_weights)
        theta = [theta[a] for a in A]

        # Step c: MCMC step (Gaussian random-walk Metropolis kernel)
        cov = kappa * np.cov(np.array(theta).T)
        for i in range(N):
            proposal = theta[i] + np.random.multivariate_normal(np.zeros(len(theta[0])), cov)
            accept_ratio = min(1, target_distribution(proposal) / target_distribution(theta[i]))
            if np.random.uniform() < accept_ratio:
                theta[i] = proposal

    
        # Step d: Update normalizing constant Z
        Z *= np.mean(w)
        results.append((theta.copy(), Z))
 
    return results
    