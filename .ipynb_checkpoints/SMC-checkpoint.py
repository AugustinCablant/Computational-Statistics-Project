""" Sequential Monte Carlo (SMC) algorithm for Bayesian inference. """
import numpy as np

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
    