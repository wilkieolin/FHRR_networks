import jax
import jax.numpy as jnp
from jax import random, jit, vmap


def phase_to_train(x, period: float = 1.0, repeats: int = 3):
    """
    Given a series of input phases defined as a real tensor, convert these values to a 
    temporal spike train. 
    """ 
    t_phase0 = period/2.0
    shape = x.shape
    
    x = x.ravel()

    #list and repeat the index 
    inds = jnp.nonzero(x)

    #list the time offset for each index and repeat it for repeats cycles
    times = x[inds]
    order = jnp.argsort(times)
    
    #sort by time
    times = times[order]
    inds = [jnp.take(inds[0], order)]
    
    n_t = times.shape[0]
    dtype = x.dtype
    
    times = times * t_phase0 + t_phase0

    #tile across time
    inds = [jnp.tile(inds[0], (repeats))]
    times = jnp.tile(times, (repeats))
        
    #create a list of time offsets to move spikes forward by T for repetitions
    offsets = jnp.arange(0, repeats, dtype=dtype) * period
    offsets = jnp.repeat(offsets, n_t)

    times += offsets
    
    return (inds, times, shape)

def current(x, t, box = 0.03):
    inds, times, full_shape = x
    
    currents = jnp.zeros(full_shape)
    
    #get the times in the current window
    cond = lambda x: (x > t - box) * (x < t + box)
    active = jnp.nonzero(cond(times))
    #swap the time indices for what flattened neuron they refer to
    active = inds[0][active]
    active = jnp.unravel_index(active, full_shape)

    currents = currents.at[active].add(1.0)
    
    return currents
    