import jax
import jax.numpy as jnp
from jax import random, jit, vmap


def phase_to_train(x, shape, period=1.0, repeats=3):
    """
    Given a series of input phases defined as a real tensor, convert these values to a 
    temporal spike train. 
    """
    n_batch = x.shape[0]
    features = jnp.prod(shape)

    output = []

    for b in range(n_batch):
        t_phase0 = period/2.0

        #create the list of indices with arbitrary dimension
        inds = jnp.arange(0, features, delta=1)
        inds = jnp.tile(inds, [repeats])
        inds = jnp.unravel_index(inds, shape)
        
        #list the time offset for each index and repeat it for repeats cycles
        times = x[b,...] * t_phase0 + t_phase0
        times = jnp.reshape(times, (-1))
        times = jnp.tile(times, [repeats])
        
        #create a list of time offsets to move spikes forward by T for repetitions
        offsets = jnp.arange(0, repeats, delta=1, dtype="float")
        offsets = jnp.repeat(offsets, features, axis=0)
        offsets = jnp.reshape(offsets, (-1)) * jnp.constant(period)
        
        times += offsets
        
        output.append( (inds, times) )
    
    return output