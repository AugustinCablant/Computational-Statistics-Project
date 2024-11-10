import numpy as np 

def SquaredBures_metric(sigma0, sigma1):
    """ 
    Compute the squared Bures metric between two covariance matrices.

    The squared Bures metric, often used in optimal transport and quantum information theory,
    provides a measure of similarity between two covariance matrices, representing the "distance"
    between Gaussian distributions with zero mean and covariance matrices sigma0 and sigma1.

    Parameters
    ----------
    sigma0 : array-like, shape (d, d)
        Covariance matrix of the first distribution. Must be symmetric and positive semi-definite.

    sigma1 : array-like, shape (d, d)
        Covariance matrix of the second distribution. Must be symmetric and positive semi-definite.

    Returns
    -------
    float
        The squared Bures metric between the two covariance matrices.
    """
    # Check if the matrices have the same shape
    assert sigma0.shape == sigma1.shape, "The covariance matrices sigma0 and sigma1 must have the same shape"
    
    # Check if the matrices are symmetric
    assert np.allclose(sigma0, sigma0.T), "The covariance matrix sigma0 is not symmetric"
    assert np.allclose(sigma1, sigma1.T), "The covariance matrix sigma1 is not symmetric"

    # Compute the square root of the covariance matrices
    sqrt_sigma0 = np.linalg.cholesky(sigma0)
    sqrt_sigma_prod = np.linalg.cholesky(np.dot(sqrt_sigma0, np.dot(sigma1, sqrt_sigma0)))

    return np.trace(sigma0 + sigma1 - 2 * sqrt_sigma_prod)


def W2(m0, sigma0, m1, sigma1):
    """
    Compute the squared Wasserstein-2 distance between two Gaussian distributions.

    The Wasserstein-2 distance, or W2 distance, between two Gaussian distributions
    is given by the sum of the squared Euclidean distance between the means and
    the Bures metric (or Wasserstein distance) between the covariance matrices.

    Parameters
    ----------
    m0 : array-like, shape (d,)
        Mean vector of the first Gaussian distribution.

    sigma0 : array-like, shape (d, d)
        Covariance matrix of the first Gaussian distribution, assumed to be symmetric and positive semi-definite.

    m1 : array-like, shape (d,)
        Mean vector of the second Gaussian distribution.

    sigma1 : array-like, shape (d, d)
        Covariance matrix of the second Gaussian distribution, assumed to be symmetric and positive semi-definite.

    Returns
    -------
    float
        The squared Wasserstein-2 distance between the two Gaussian distributions.
    """
    return np.linalg.norm(m0 - m1) ** 2 + SquaredBures_metric(sigma0, sigma1)

def KL_divergence(P, Q, epsilon = 1e-10):
    """
    Compute the KL divergence between two distributions

    Parameters
    ----------
    P : array-like
        First distribution.
    Q : array-like
        Second distribution.
    epsilon : float, optional
        Small value to avoid division by zero, by default 1e-10.
    """
    assert len(P) == len(Q), "The two distributions must have the same length"

    # Avoid division by zero and log(0) by adding epsilon
    P = np.where(P == 0, 0, P)
    Q = np.where(Q == 0, epsilon, Q)

    return np.sum(P * np.log(P / Q))