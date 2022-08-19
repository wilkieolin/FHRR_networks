"""
This file contains the classes used to create vector-symbolic priors. 

Contains code adapted from the Deepmind Perceiver model under the Apache
2.0 License.

Wilkie Olin-Ammentorp, 2022
University of Califonia, San Diego
"""

import abc

import haiku as hk
import jax
import jax.numpy as jnp

class AbstractPositionEncoding(hk.Module, metaclass=abc.ABCMeta):
    """Abstract Perceiver decoder."""

    @abc.abstractmethod
    def __call__(self, batch_size, pos):
        raise NotImplementedError


class TrainablePositionEncoding(AbstractPositionEncoding):
    """Trainable VSA position encoding."""

    def __init__(self, n_encodings, vsa_dimension, name=None):
        super(TrainablePositionEncoding, self).__init__(name=name)
        self._n_encodings = n_encodings
        self._vsa_dimension = vsa_dimension

    def __call__(self, n_batch):
        pos_embs = hk.get_parameter(
            'pos_embs', [self._n_encodings, self._vsa_dimension],
            init=hk.initializers.RandomUniform(minval = -1.0, maxval = 1.0))

        pos_embs = jnp.broadcast_to(pos_embs, (n_batch, self._n_encodings, self._vsa_dimension))

        return pos_embs
