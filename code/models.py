# Need both Jax and Tensorflow Probability 
import jax
# Important to enable 64-bit precision
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random


def skewgaussian(t, params):
    """
    calculates a skew-Gaussian function at times `t` for a given 
    set of parameters stored in `params`.

    Parameters
    ----------
    t : numpy.ndarray or JAX array
        Set of time stamps at which to evaluate the 
        skew-Gaussian function
        
    params : iterable
        list of parameters for the skew-Gaussian. They are:
            * `logA`: log-amplitude (height)
            * `t0`: time of peak
            * `logsig1`: log of the rise timescale
            * `logsig2`: log of the fall timescale
    
    Returns
    -------
    y : numpy.ndarray or JAX array
        Array with skew-Gaussian evaluated at times `t`
    """
    logA = params[0]
    t0 = params[1]
    logsig1 = params[2]
    logsig2 = params[3]

    y = jnp.exp(logA) * jnp.where(
            t > t0,
            jnp.exp(-((t - t0) ** 2) / (2 * (jnp.exp(logsig2)**2))),
            jnp.exp(-((t - t0) ** 2) / (2 * (jnp.exp(logsig1)**2))),
        )
    return y
