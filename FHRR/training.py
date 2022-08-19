"""
This file contains methods used to load data and train example network models.

Wilkie Olin-Ammentorp, 2022
University of Califonia, San Diego
"""

import haiku as hk
import jax
import jax.numpy as jnp

import optax
import tensorflow_datasets as tfds

from FHRR.modules import *
from FHRR.utils import *
from tqdm import tqdm
from functools import reduce

#
# Data functions
#

def load_dataset(dataset_name: str,
                split: str,
                *,
                is_training: bool,
                batch_size: int,
                repeat: bool = True):
    """Loads the dataset as a generator of batches."""


    ds = tfds.load(dataset_name, data_dir="~/data", split=split).cache()
    if repeat:
        ds = ds.repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)

    x_full, y_full = tfds.as_numpy(tfds.load(dataset_name, 
                    split=[split], 
                    data_dir="~/data",
                    shuffle_files=True,
                    as_supervised=True,
                    batch_size=-1,
                    with_info=False))[0]

    return iter(tfds.as_numpy(ds)), x_full, y_full

def scale_mnist(images):
    """Convert int8 based values to normalized floats"""
    return jnp.divide(images, 255)

#
# Model functions
#


def accuracy(net, key, params, images, labels, **kwargs):
    """
    Compute classification accuracy given a model, parameters, and dataset.
    """

    yhat = net.apply(params, key, images, **kwargs)
    yhat = jnp.argmax(yhat, axis=1)
    return yhat == labels


def update_params(model: hk.Transformed, 
                key,
                trainable_params: hk.Params = None, 
                non_trainable_params: hk.Params = None, 
                loss_fn = None, 
                data = None, 
                optimizer = None, 
                opt_state = None, 
                **kwargs):
    """
    Training loop update step. 
    """

    #separate the image and label data from the batch
    xd = data['image']
    yd = data['label']

    #lambda function to compute loss 
    batch_loss = lambda tp, ntp: jnp.mean(loss_fn(
                                        model.apply(hk.data_structures.merge(tp, ntp), key, xd, is_training=True, **kwargs),
                                        yd))

    #compute the loss value and gradients
    loss_value = batch_loss(trainable_params, non_trainable_params)
    gradients = jax.grad(batch_loss)(trainable_params, non_trainable_params)
    
    #compute trainable parameter updates using the optimizer
    updates, opt_state = optimizer.update(gradients, opt_state)
    new_trainable_params = optax.apply_updates(trainable_params, updates)
    
    return new_trainable_params, opt_state, loss_value

def train_model(model, 
                key,
                params = None,
                optimizer = None,
                dataset = None,
                loss_fn = None,
                batches: int = None,
                loss_history = None,
                opt_state = None,
                non_trainable_params = ["codebook", "static_projection", "classification_query"],
                **kwargs):
    """
    Main training loop for reducing loss for a model on a dataset.
    """

    #separate the trainable and non-trainable model parameters - non-trainable parameters are passed
    # via the non_trainable_params arg above
    trainable_params, non_trainable_params = hk.data_structures.partition(
        #return false for each of the non-trainable parameters in the model
        lambda m, n, v: bool(reduce(lambda a,b: a*b, [name not in m for name in non_trainable_params])),
        params
    )

    #if there is no loss history to append to (first run)
    if loss_history == None:
        loss_history = []

    #if there is no previous optimizer state (first run or stateless optimizer)
    if opt_state == None:
        opt_state = optimizer.init(trainable_params)

        
    #lambda calls parameters, batch, and optimizer state
    update_fn = lambda train_params, nontrain_params, batch, opt_state: update_params(
                            model, 
                            key, 
                            trainable_params = train_params, 
                            non_trainable_params = nontrain_params, 
                            loss_fn = loss_fn, 
                            data = batch, 
                            optimizer = optimizer, 
                            opt_state = opt_state, 
                            **kwargs)

    #main optimizer loop
    for i in tqdm(range(batches)):
        batch = next(dataset)
        #call the update lambda
        trainable_params, opt_state, loss_val = update_fn(trainable_params, 
                                                    non_trainable_params, 
                                                    batch, 
                                                    opt_state)

        #append the loss value
        loss_history.append(loss_val)

    #merge the parameters back together
    new_params = hk.data_structures.merge(trainable_params, non_trainable_params)

    return new_params, loss_history

