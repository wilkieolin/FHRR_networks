"""
This file contains the methods used for computing FHRR functions and other basic
utilities. 

Contains code adapted from the Deepmind Perceiver model and deepmind-haiku under 
the Apache 2.0 License.

Wilkie Olin-Ammentorp, 2022
University of Califonia, San Diego
"""

import jax
import jax.numpy as jnp
import haiku as hk
import tensorflow as tf
from jax import random, jit, vmap
from einops import rearrange
from optax import NonNegativeParamsState
from typing import Any, Sequence


def attend(queries, keys, values, dropout_prob: float = 0.0, attention_mask = NonNegativeParamsState):
    """
    Compute attention between vector-symbolic query, key, and value.

    queries, keys, values - (b x d) symbol matrix
    dropout_prob - float [0.0, 1.0], determines probability of score dropout
    attention_mask - mask over scores to use
    """

    """
    einsum dimensions below:
    b - batch
    h - head
    t / T - inputs / "time"
    d - VSA dimension
    """
    
    n_batch, n_heads, _, _ = queries.shape
    
    #rearrange queries and keys so we can dispatch over dimension 0
    q_reshaped = rearrange(queries, "b h t d -> (b h) t d")
    k_reshaped = rearrange(keys, "b h T d -> (b h) T d")
    
    #get the similarity scores between Q and K by batch and head
    #scores is (b h t T)
    scores = vmap(similarity_outer)(q_reshaped, k_reshaped)
    
    #dropout scores
    if dropout_prob > 0.0:
        scores = hk.dropout(hk.next_rng_key(), dropout_prob, scores)

    if attention_mask is not None:
        #with VSA attention you don't have to go through softmax so set masked elements
        #directly to zero
        zeros = jnp.array(0.0)
        scores = jnp.where(attention_mask, scores, zeros)
        
    #do complex-domain matrix multiplication of scores by values
    scores = jnp.complex64(scores)
    scores = rearrange(scores, "(b h) t T -> b h t T", b = n_batch, h = n_heads)
    values = unitary_to_cmpx(values)
    output = jnp.einsum("b h t T, b h T d-> b h t d", scores, values)
    output = cmpx_to_unitary(output)
    
    #output is (b h t d)

    return output

def bind(symbols):
    """
    FHRR binding operation (x)

    (2 d) inputs -> (d) output
    Given matrix of two symbols (2 d), sum angles along the zeroth dimension and remap to (-1,1)
    """
    #sum the angles
    symbol = jnp.sum(symbols, axis=0)
    #remap the angles to (-1, 1)
    symbol = remap_phase(symbol)

    return symbol

def bind_list(*symbols):
    """
    FHRR list binding operation (x)

    [(arbitrary), (arbitrary)] inputs -> (arbitrary) output
    Given two tensors of symbols, stack along the zeroth dimension and pass to the bind op
    """
    #stack the vararg inputs into an array
    symbols = jnp.stack(symbols, axis=0)

    return bind(symbols)

def bundle(symbols, n=-1):
    """
    FHRR bundling operation (+)

    (n d) inputs -> (1 d) output
    Given a list of symbols (n d) find one output symbol (1 d) which is as similar
    as possible to the input list (complex average). 

    The first symbol in the list can be 'weighted' to change its impact on the average.
    """

    #sum the complex numbers to find the bundled vector
    cmpx = unitary_to_cmpx(symbols)
    if n > 1:
        cmpx_0 = n * cmpx[0:1, :]
        cmpx_1 = cmpx[1:, :]
        cmpx = jnp.stack((cmpx_0, cmpx_1), axis=0)

    bundle = jnp.sum(cmpx, axis=0)
    #convert the complex sum back to an angle
    bundle = cmpx_to_unitary(bundle)
    bundle = jnp.reshape(bundle, (1, -1))

    return bundle

def bundle_list(*symbols):
    """
    FHRR list bundling operation (+)

    [(arbitrary)...] inputs -> (arbitrary) output
    Given tensors of symbols, stack along the zeroth dimension and pass to the bundling op
    """
    #stack the vararg inputs into an array
    symbols = jnp.stack(symbols, axis=0)

    return bundle(symbols)

def cmpx_to_unitary(cmpx):
    """
    Convert complex numbers back to radian-normalized angles on the domain (-1, 1)
    """

    pi = jnp.pi
    #convert the complex sum back to an angle
    symbol = jnp.angle(cmpx) / pi

    return symbol

def disable_gpu_tf():
    """
    Disable visible devices to tensorflow so it and JAX don't fight over memory
    """

    tf.config.set_visible_devices([], 'GPU')


def generate_symbols(key, number: int, dimensionality: int):
    """
    Helper alias to generate FHRR symbols from jax.random.uniform 

    key: JAX PRNG key
    number: int, number of symbols to generate
    dimensionality: int, VSA symbol dimensionality
    """

    return random.uniform(key, minval=-1.0, maxval=1.0, shape=(number, dimensionality))


@jax.jit
def onehot_loss(similarities, labels):
    """
    Given a list of outer similarities between symbols (b c) and a one-hot list of labels (b c),
    return the similarity for only the correct class label.

    similarities: (b c) matrix of similarities between a symbol and class codebook
    labels: (b) vector of class labels
    """

    loss = 1.0 - vmap(lambda x, y: x[y])(similarities, labels)

    return loss

@jax.jit
def phasor_act(z):
    """
    Second part of the phasor activation function: given a complex number, return its
    radian-normalized angle. 
    """

    pi = jnp.pi
    return jnp.angle(z) / pi
    

def normalize_PCM16(x):
    """
    Normalize a PCM16-based signal
    """

    sig_16_max = 2 ** 15 - 1
    return x / sig_16_max
    

def remap_phase(x):
    """
    Remap radian-normalized angles from (-inf, inf) to equivalent (-1, 1) angles
    """
    x = jnp.mod(x, 2.0)
    x = -2.0 * jnp.greater(x, 1.0) + x

    return x

@jit
def similarity(a,b):
    """
    FHRR symbol similarity operation

    (b d), (b d) or (1 d), (b d) inputs -> (b) outputs
    Similarity of vectors is defined by the cosine of the difference between each
    corresponding angle in two symbols, averaged
    """
    #multiply values by pi to move from (-1, 1) to (-π, π)
    pi = jnp.pi
    a = jnp.multiply(a, pi)
    b = jnp.multiply(b, pi)

    #calculate the mean cosine similarity between the vectors
    similarity = jnp.mean(jnp.cos(a - b), axis=1)

    return similarity

def similarity_outer(a,b):
    """
    FHRR symbol similarity between two sets of vectors
    (b d), (c d) inputs -> (b c) outputs
    """

    sim_op = lambda x: similarity(x, b)
    
    return vmap(sim_op)(a)

class SparseRandomUniform(hk.initializers.Initializer):
    """
    Constructs a :class:`SparseRandomUniform` initializer.

    minval: The lower limit of the uniform distribution.
    maxval: The upper limit of the uniform distribution.
    sparsity: The proportion of values to mask with zero.
    """

    def __init__(self, minval=0., maxval=1., sparsity=0.0):
        

        self.minval = minval
        self.maxval = maxval
        self.sparsity = sparsity

    def __call__(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:
        param =  jax.random.uniform(hk.next_rng_key(), shape, dtype, self.minval, self.maxval)

        if self.sparsity > 0.0:
            mask = jax.random.uniform(hk.next_rng_key(), shape, dtype, 0.0, 1.0) > self.sparsity
            param = param * mask

        return param


def unitary_to_cmpx(symbols):
    """
    Convert phasor symbols on the domain (-1, 1) to complex numbers
    lying on the unit circle
    """

    #convert each angle to a complex number
    pi = jnp.pi
    j = jnp.array([0+1j])
    #sum the complex numbers to find the bundled vector
    cmpx = jnp.exp(pi * j * symbols)

    return cmpx

def unbundle(x, symbols, n=-1):
    """
    FHRR unbundling operation

    (1 d), (n d) inputs -> (1 d) output
    number of symbols bundled to form x is assumed from 'symbols' matrix or
    passed manually by n

    note - this operation is highly approximate
    """

    #assume that the number of symbols bundled to form x is the number
    #passed in the matrix plus one
    if n <= 0:
        n = symbols.shape[0] + 1
    assert n >= symbols.shape[0], "Too many symbols to unbundle given this weight"
    
    x_cmpx = unitary_to_cmpx(x) * n
    s_cmpx = unitary_to_cmpx(symbols)

    symbol = x_cmpx - jnp.sum(s_cmpx, axis=0)
    symbol = cmpx_to_unitary(symbol)
    symbol = jnp.reshape(symbol, (1, -1))

    return symbol

def unbind(x, symbols):
    """
    FHRR unbinding operation

    (1 d), (n d) inputs -> (1 d) output
    """
    symbols = jnp.sum(symbols, axis=0)

    #remove them from the input & remap phase
    symbol = jnp.subtract(x, symbols)
    symbol = remap_phase(symbol)

    return symbol

def unbind_list(x, *symbols):
    """
    FHRR list unbinding operation
    (1 d), [(1 d) ...] inputs -> (1 d) output
    """

    #stack and sum the symbols to be unbound
    symbols = jnp.stack(symbols, axis=0)

    return unbind(x, symbols)
