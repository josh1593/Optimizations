# this file contains collections of proxes we learned in the class
import numpy as np
from scipy.optimize import bisect

# =============================================================================
# TODO Complete the following prox for simplex
# =============================================================================

# Prox of capped simplex
# -----------------------------------------------------------------------------


def prox_csimplex(z, k):
    """
    Prox of capped simplex
            argmin_x 1/2||x - z||^2 s.t. x in k-capped-simplex.

    input
    -----
    z : arraylike
            reference point
    k : int
            positive number between 0 and z.size, denote simplex cap

    output
    ------
    x : arraylike
            projection of z onto the k-capped simplex
    """
    # safe guard for k
    assert 0 <= k <= z.size, 'k: k must be between 0 and dimension of the input.'

    # TODO do the computation here
    # Hint: 1. construct the scalar dual object and use `bisect` to solve it.
    # 2. obtain primal variable from optimal dual solution and return it.
    #
    def dual_derivative(y, z, k):
       return np.sum(np.clip(z - y, 0, 1)) - k

    def x(z, y):
       return np.clip(z - y, 0, 1)

    from scipy.optimize import bisect
    y = bisect(dual_derivative, -np.sum(np.abs(z)), np.sum(np.abs(z)), args=(z,k))
    return x(z, y)



def prox_l1(x, t):
    """
    regular l1 prox included for convenience
    Note that you'll have to rescale the t input with the regularization parameter
    """
    y = np.zeros(x.size)
    ind = np.where(np.abs(x) > t)
    x_o = x[ind]
    y[ind] = np.sign(x_o)*(np.abs(x_o) - t)
    return y


def rank_project(Y, k):
    """	Prox of rank constrained matrices
            argmin_M 1/2||M - Y||^2 s.t. rank(M)<=k


    Parameters
    ----------
    Y : 2 dimensional array
    k : positive integer

    Returns
    -------
    2 dimensional array
            rank projected version of Y
    """
    # TODO write this function
    U_rank, sigma_rank, VT_rank = np.linalg.svd(Y)
    for i in range(len(sigma_rank)):
        if i >= k:
            sigma_rank[i] = 0
        else: 
            sigma_rank[i] = sigma_rank[i]
            
    return U_rank @ np.diag(sigma_rank) @ VT_rank
    pass


def nuclear_prox(Y, t):
    """Nuclear norm proximal operator
    argmin_M 1/(2t)||M - Y||^2 + ||M||_{*}

    Parameters
    ----------
    Y : 2 dimensional array
    k : positive integer

    Returns
    -------
    2 dimensional array
            proximal operator applied to Y
    """
    # TODO write this function
    U_prox, sigma_prox, VT_prox = np.linalg.svd(Y)
    return U_prox @ np.diag(prox_l1(sigma_prox,t)) @ VT_prox

    pass
