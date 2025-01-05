""" Sequential Monte Carlo (SMC) algorithm for Bayesian inference. """

import numpy as np

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

def Tempering_SMC(N, tau, rw):
    """ 
    Perform the Tempering Sequential Monte Carlo (SMC) algorithm for Bayesian inference.

    The Tempering SMC algorithm is a variant of the SMC algorithm that uses a sequence of
    intermediate distributions to sample from the target distribution. This sequence of
    intermediate distributions is defined by a sequence of temperatures, which are used to
    interpolate between the prior distribution and the target distribution.

    Parameters
    ----------
    N : int
        Number of particles to use in the SMC algorithm.

    tau : float
         ESS threshold for resampling.

    rw : positive
        random walk tuning parameter.

    Returns
    -------
    array-like, shape (N,)
        Array of samples from the target distribution.
    """
    
    t = 1
    lambda_0 = 0
    Z_0 = 1
    theta_0 = []
    