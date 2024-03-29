"""
This file contains neural network layers and models used to build networks.

Contains code adapted from the Deepmind Perceiver model under the Apache
2.0 License.

Wilkie Olin-Ammentorp, 2022
University of Califonia, San Diego
"""

import jax
from jax.lax import complex
import jax.numpy as jnp
import haiku as hk
from einops import rearrange
from FHRR.utils import *
from FHRR.spiking import *


#   _____      _           _ _   _                
#  |  __ \    (_)         (_) | (_)               
#  | |__) | __ _ _ __ ___  _| |_ ___   _____  ___ 
#  |  ___/ '__| | '_ ` _ \| | __| \ \ / / _ \/ __|
#  | |   | |  | | | | | | | | |_| |\ V /  __/\__ \
#  |_|   |_|  |_|_| |_| |_|_|\__|_| \_/ \___||___/


class CodebookDecoder(hk.Module):
    """
    Codebook decoder, stores a set of symbols and returns the outer similarity
    between the input and stored symbols. Used for classification, etc. 

    Note - if passed in a parameter collection, separate this into non-trainable
    parameters during training loop. (Done by default in train_model)
    """

    def __init__(self, n_codes,channel_size, sparsity: float = 0.0, name=None):
        super().__init__(name=name)
        self.n_codes = n_codes
        self.channel_size = channel_size
        self.sparsity = sparsity
        
    def __call__(self, x):
        codebook = hk.get_parameter("codebook", 
                                    shape = [self.n_codes, self.channel_size], 
                                    dtype = x.dtype, 
                                    init = SparseRandomUniform(minval=-1.0, maxval=1.0, sparsity=self.sparsity))

        return similarity_outer(x, codebook)

class GraphEncoder(hk.Module):
    """
    Given a graph in the Cardiotox dataset, encode it into a VSA.
    
    dim_vsa: int, length of vector-symbol used in the architecture
    dim_node: number of features used to describe an node (atom)
    dim_edge: number of features used to describe an edge (bond)
    max_edges: maximum number of edges (bonds) in the dataset being encoded
    """

    def __init__(self, dim_vsa: int, dim_node: int = 27, dim_edge: int = 12, max_edges: int = 70, name=None):
        super().__init__(name=name)
        self.dim_vsa = dim_vsa
        self.dim_edge = dim_edge
        self.dim_node = dim_node
        self.max_edges = max_edges
        
    def __call__(self, x, ):
        #get batch size
        active = x['active']
        n_batch = active.shape[0]
        
        graphs = []
        
        
        def project_node(atoms):
            atom_projection = hk.get_parameter("atom_projection", 
                                        shape = [self.dim_node, self.dim_vsa],
                                        init = hk.initializers.RandomUniform())
            
            atoms = np.einsum("x a, a d -> x d", atoms, atom_projection)
            atoms = remap_phase(atoms)
            return atoms
            
        def project_edge(edges):
            bond_projection = hk.get_parameter("bond_projection", 
                                        shape = [self.dim_edge, self.dim_vsa],
                                        init = hk.initializers.RandomUniform())
            
            edges = np.einsum("x a, a d -> x d", edges, bond_projection)
            edges = remap_phase(edges)
            return edges
            
        for i in range(n_batch):
            #load the active bonds (edges)
            #undirected graph so only use upper triangle of symmetric adj. matrix
            active_edges = tf.experimental.numpy.triu(x['pair_mask'][i,...])
            active_edges = tf.where(active_edges)
            edges = tf.gather_nd(x['pairs'][i], active_edges)
            
            #find the atoms for each bond
            sources = active_edges[:,0]
            source_atoms = tf.gather(x['atoms'][i], sources, axis=0)
            
            destinations = active_edges[:,1]
            dest_atoms = tf.gather(x['atoms'][i], destinations, axis=0)
            
            #convert to device array
            source_atoms = jnp.array(source_atoms)
            dest_atoms = jnp.array(dest_atoms)
            edges = jnp.array(edges)
            
            #project each node into its symbol
            source_atoms = project_node(source_atoms)
            dest_atoms = project_node(dest_atoms)
            
            #project each edge into its symbol
            edges = project_edge(edges)
            
            #add the spectrum (frequency encoding) to each node
            n_edges = edges.shape[0]
            f_basis = hk.get_parameter("frequency_basis",
                                      shape = [1, self.dim_vsa],
                                      init = hk.initializers.RandomUniform())
            
            spectrum = generate_spectrum(f_basis, n_edges)

            #construct the graph from atoms, edges, and bundle with the spectrum
            graph = bind_list(source_atoms, dest_atoms, edges)
            graph = bundle_list(graph, spectrum)
            
            #pad out the array
            
            padding = np.zeros((self.max_edges - n_edges, self.dim_vsa))
            graph = np.concatenate((graph, padding), axis=0)
            #move back to CPU
            graph = np.array(graph)
            
            graphs.append(graph)
            
        return np.stack(graphs)

class PhasorDense(hk.Module):
    """
    Fully-connected / "Linear" layer which implements phasor activation.

    output_size: int, number of neurons in layer
    w_init: layer weight initializer
    mask_angle: float, half-angle of arc subtended from 0* which masks inputs. used for sparse layers.
    """

    def __init__(self, output_size, w_init, mask_angle: float = -1.0, name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.w_init = w_init
        self.mask_angle = mask_angle
        
    def __call__(self, x, spiking: bool = False, **kwargs):
        if spiking:
            return self.call_dynamic(x, **kwargs)

        #access the weights / biases
        j, k = x.shape[-1], self.output_size
        w = hk.get_parameter("w", shape=[j, k], dtype="float", init=self.w_init)
        b = hk.get_parameter("bz", shape=[k], dtype="float", init=jnp.ones)

        #convert the phase angles to complex numbers
        pi = jnp.pi
        imag = complex(0.0, 1.0)
        xz = jnp.exp(imag * pi * x)
        #mask all inputs inside the arc of the mask angle
        if self.mask_angle > 0.0:
            mask = jnp.greater_equal(jnp.abs(x), self.mask_angle)
            xz = xz * mask
        
        #convert weights to complex
        wz = complex(w, jnp.zeros_like(w))
        bz = complex(b, jnp.zeros_like(b))
        
        #do the complex sum & take the angle
        z = jnp.matmul(xz, wz) + bz
        y = phasor_act(z)
        #mask all outputs inside the arc of the mask angle
        if self.mask_angle > 0.0:
            mask = jnp.greater_equal(jnp.abs(y), self.mask_angle)
            y = y * mask
        
        return y

    def call_dynamic(self, 
                        x, 
                        t_box: float = 0.03, 
                        t_step: float = 0.01, 
                        t_range = (0.0, 10.0), 
                        z_init = None,
                        threshold: float = 0.05,
                        gpu: bool = True,
                        offset: float = 0.0,
                        **kwargs):
        
        indices, times, full_shape = x

        #access the weights / biases
        n_batch, n_input = full_shape
        n_output = self.output_size
        w = hk.get_parameter("w", shape=[n_input, n_output], dtype="float", init=self.w_init)
        bz = hk.get_parameter("bz", shape=[n_output], dtype="float", init=jnp.ones)

        #define the initial state
        state_shape = (n_batch, n_output)
        if z_init is None:
            z_init = np.zeros(state_shape, dtype="complex128")
        else:
            assert z_init.shape is state_shape, "Initial z-values must match batch & layer shape."

        #define the current-generating function
        t_grid = define_tgrid(t_range, t_step)
        active_inds = generate_active(x, t_grid, t_box)
        current_fn = lambda t: current(x, active_inds, t, t_step)
        #define the differential update
        if gpu:
            dz_fn = lambda t, z: dz_dt_gpu(current_fn, t, z, weight=w, bias=bz, **kwargs)
        else:
            dz_fn = lambda t, z: dz_dt(current_fn, t, z, weight=w, bias=bz, **kwargs)

        #integrate through time
        if self.name is not None:
            print("Solving layer", self.name)

        solution = solve_heun(dz_fn, t_grid, t_step, z_init)

        #find and return the spikes produced
        y = find_spikes(solution, threshold=threshold)

        #exclude spikes from the inhibitory period if mask angle is being used
        if self.mask_angle > 0.0:
            y = inhibit_midpoint(y, self.mask_angle, 1.0, offset=offset)

        return y


class PhasorMultiDense(hk.Module):
    """
    Stack of linear layers to provide VSA eqivalent of multi-head attention.

    output_size: int, number of neurons in layer
    w_init: layer weight initializer
    n_heads: int, number of layers in stack
    mask_angle: float, half-angle of arc subtended from 0* which masks inputs. used for sparse layers.
    """

    def __init__(self, output_size, w_init, n_heads:int = 1, mask_angle:float = -1.0, name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.w_init = w_init
        self.n_heads = n_heads
        self.n_neurons = self.output_size * self.n_heads
        self.mask_angle = mask_angle
        
    def __call__(self, x):
        n_batch, n_time, n_dim = x.shape

        #access the weights / biases
        weight_shape = [self.n_heads, n_dim, self.output_size]
        w = hk.get_parameter("w", shape=weight_shape, dtype=x.dtype, init=self.w_init)

        bias_shape = [self.n_heads, self.output_size]
        bz = hk.get_parameter("bz", shape=bias_shape, dtype="complex64", init=jnp.ones)
        
        #flatten batch and time
        x = rearrange(x, "b t d -> (b t) d")
        #convert the phase angles to complex numbers
        pi = jnp.pi
        imag = jnp.complex64(1j)
        xz = jnp.exp(imag * pi * x)
        #mask all inputs inside the arc of the mask angle
        if self.mask_angle > 0.0:
            mask = jnp.greater_equal(jnp.abs(x), self.mask_angle)
            xz = xz * mask
        
        #convert weights to complex
        wz = jnp.complex64(w)
        
        #do the complex sum & take the angle
        yz = jnp.einsum("x d, h d D -> x h D", xz, wz) + bz
        y = phasor_act(yz)
        #reshape batch and time
        y = rearrange(y, "(b t) h d -> b h t d", b = n_batch, t = n_time)
        #mask all outputs inside the arc of the mask angle
        if self.mask_angle > 0.0:
            mask = jnp.greater_equal(jnp.abs(y), self.mask_angle)
            y = y * mask

        return y

class ProjectRow(hk.Module):
    """
    Given batch of 3-D inputs, collapse the last two dimensions and project them into a symbol.

    (b x y c) -> (b x d)
    dim_vsa: int, length of vector-symbol used in the architecture
    sigma: float, layerNorm scaling. 3.0 should keep ~99% of inputs from clipping over [-1,1]
    """

    def __init__(self, dim_vsa: int, sigma: float = 3.0, name=None):
        super().__init__(name=name)
        self.dim_vsa = dim_vsa
        self.sigma = sigma

    def __call__(self, x):
        n_batch, n_rows, n_columns, n_channels = x.shape
        
        n_in = n_columns * n_channels
        random_projection = hk.get_parameter("static_projection", 
                                    shape = [n_in, self.dim_vsa],
                                    init = hk.initializers.RandomNormal())
        
        
        x = rearrange(x, "b x y c -> b x (y c)")
        x = jnp.einsum("b x a, a d -> b x d", x, random_projection)
        x = layer_norm(x) / self.sigma
        
        return x

class ProjectAll(hk.Module):
    """
    Given batch of 3-D inputs, collapse the last three dimensions and project them into a symbol.

    (b x y c) -> (b d)
    dim_vsa: int, length of vector-symbol used in the architecture
    sigma: float, layerNorm scaling. 3.0 should keep ~99% of inputs from clipping over [-1,1]
    """
    def __init__(self, dim_vsa, sigma=3.0, name=None):
        super().__init__(name=name)
        self.dim_vsa = dim_vsa
        self.sigma = sigma

    def __call__(self, x):
        n_batch, n_rows, n_columns, n_channels = x.shape
        
        n_in = n_columns * n_channels * n_rows
        random_projection = hk.get_parameter("static_projection", 
                                    shape = [n_in, self.dim_vsa],
                                    init = hk.initializers.RandomNormal())
        
        
        x = rearrange(x, "b x y c -> b (x y c)")
        x = jnp.einsum("b a, a d -> b d", x, random_projection)
        x = layer_norm(x) / self.sigma
        
        return x

def conv_1d(dimension, init_scale=1.0):
    """
    Convenience wrapper around PhasorDense call
    """
    return PhasorDense(dimension, 
                     w_init=hk.initializers.VarianceScaling(init_scale))

def conv_1d_mh(dimension, n_heads, init_scale=1.0):
    """
    Convenience wrapper around PhasorMultiDense call
    """
 
    return PhasorMultiDense(dimension, 
                     n_heads = n_heads,
                     w_init=hk.initializers.VarianceScaling(init_scale))

def layer_norm(x, name=None):
    """
    Convenience wrapper around LayerNorm call
    """
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)

def fhrr_norm(x, name=None, sigma=3.0):
    """
    Convenience wrapper around fhrr_norm call
    """
    x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)(x)
    return jnp.divide(x, sigma)


class MLP(hk.Module):
    """
    MLP layer to follow attention block

    widening_factor: int > 1, how many times wider than the symbol hidden layer will be
    dropout_prob: float [0.0, 1.0], dropout probability on output
    """

    def __init__(self,
                widening_factor = 4,
                dropout_prob = 0.0,
                init_scale = 1.0,
                name = None,):

        super(MLP, self).__init__(name=name)
        self.widening_factor = widening_factor
        self.dropout_prob = dropout_prob
        self.init_scale = init_scale

    def __call__(self, inputs, is_training=False):
        if is_training:
            dropout = self.dropout_prob
        else:
            dropout = 0.0

        output_channels = inputs.shape[-1]
        x = conv_1d(output_channels * self.widening_factor, self.init_scale)(inputs)
        x = conv_1d(output_channels, self.init_scale)(x)
        output = hk.dropout(hk.next_rng_key(), dropout, x)

        return output

#           _   _             _   _               __  __           _       _           
#      /\  | | | |           | | (_)             |  \/  |         | |     | |          
#     /  \ | |_| |_ ___ _ __ | |_ _  ___  _ __   | \  / | ___   __| |_   _| | ___  ___ 
#    / /\ \| __| __/ _ \ '_ \| __| |/ _ \| '_ \  | |\/| |/ _ \ / _` | | | | |/ _ \/ __|
#   / ____ \ |_| ||  __/ | | | |_| | (_) | | | | | |  | | (_) | (_| | |_| | |  __/\__ \
#  /_/    \_\__|\__\___|_| |_|\__|_|\___/|_| |_| |_|  |_|\___/ \__,_|\__,_|_|\___||___/
                                                                                     
                                                                                     
class Attention(hk.Module):
    """
    Given a query input and key/value input, calculate the VSA-based attention scores
    between the inputs and scale the values by these scores. Followed by 1D convolution.
    """
    
    def __init__(self, 
                 qk_channels = None, 
                 v_channels=None, 
                 output_channels=None, 
                 init_scale = 1.0, 
                 dropout_prob = 0.0,
                 num_heads=1, 
                 sigma=3.0,
                 name=None):
        
        super(Attention, self).__init__(name=name)
        self.init_scale = init_scale
        self.n_heads = num_heads
        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.output_channels = output_channels
        self.dropout_prob = dropout_prob
        self.sigma = sigma
        
        
    def __call__(self, query, keyvalue, attention_mask = None):
        #inputs: Query - (b t d), Keyvalue - (b T d)
        #outputs: (b t d)
        
        #use the same logic in perceiver layout: q and k need same (VSA) dimension, V can take different shape
        #with a VSA attention layer we don't split over heads since it's a distributed symbol
        #we produce multiple full symbols per input instead
        
        if self.qk_channels == None:
            self.qk_channels = query.shape[-1]
        
        #by default expect the same channels in values as query
        if self.v_channels == None:
            self.v_channels = self.qk_channels
            
        #by default return the same output channels as the values
        if self.output_channels == None:
            self.output_channels = self.v_channels
        
        #produces shape (b h d) 
        # batch, heads, dims (qk/v channels)
        xform_call = lambda x: conv_1d_mh(x, n_heads=self.n_heads, init_scale=self.init_scale)
        
        q = xform_call(self.qk_channels)(query)
        k = xform_call(self.qk_channels)(keyvalue)
        v = xform_call(self.v_channels)(keyvalue)
        
        result = attend(q, k, v, dropout_prob = self.dropout_prob, attention_mask = attention_mask)
        #flatten heads out for final layer
        result_flat = rearrange(result, 'b h t d -> b t (h d)')
        
        output = PhasorDense(self.output_channels, 
                             w_init = hk.initializers.VarianceScaling(self.init_scale))(result_flat)
        
        return output    

class SelfAttention(hk.Module):
    """
    Standard self-attention module
    """

    def __init__(self,
                widening_factor = 4,
                dropout_prob = 0.0,
                dropout_attn_prob = 0.0,
                num_heads = 2,
                att_init_scale = 1.0,
                dense_init_scale = 1.0,
                qk_channels = None,
                v_channels = None,
                name = None):

        super(SelfAttention, self).__init__(name=name)
        self.widening_factor = widening_factor
        self.dropout_prob = dropout_prob
        self.dropout_attn_prob = dropout_attn_prob
        self.num_heads = num_heads
        self.att_init_scale = att_init_scale
        self.dense_init_scale = dense_init_scale
        self.qk_channels = qk_channels
        self.v_channels = v_channels

    def __call__(self,
                inputs,
                *,
                attention_mask = None,
                is_training):

        dropout_prob = self.dropout_prob if is_training else 0.0
        dropout_attn_prob = self.dropout_attn_prob if is_training else 0.0

        #save x for residual
        x = inputs

        #put the inputs through batch norm
        qkv_inputs = layer_norm(inputs)

        #self-attend over the inputs
        attention = Attention(
            num_heads=self.num_heads,
            init_scale=self.att_init_scale,
            qk_channels=self.qk_channels,
            v_channels=self.v_channels,
            dropout_prob=dropout_attn_prob,)(qkv_inputs, qkv_inputs,
                                            attention_mask=attention_mask)
        #output shape (b h t d)
        attention = hk.dropout(hk.next_rng_key(), dropout_prob, attention)
 
        #residual symbol binding on attention output
        x = bind_list(x, attention)

        x = bind_list(x, MLP(
        widening_factor=self.widening_factor,
        dropout_prob=dropout_prob,
        init_scale=self.dense_init_scale)(
            layer_norm(x), is_training=is_training))

        return x

class CrossAttention(hk.Module):
    """
    Cross-attention module used to build perceivers
    """

    def __init__(self,
                widening_factor = 1,
                dropout_prob = 0.0,
                dropout_attn_prob = 0.0,
                num_heads = 8,
                att_init_scale = 1.0,
                dense_init_scale = 1.0,
                shape_for_attn = 'kv',
                use_query_residual = True,
                qk_channels = None,
                v_channels = None,
                name = None):

        super(CrossAttention, self).__init__(name=name)
        self._widening_factor = widening_factor
        self._dropout_prob = dropout_prob
        self._dropout_attn_prob = dropout_attn_prob
        self._num_heads = num_heads
        self._att_init_scale = att_init_scale
        self._dense_init_scale = dense_init_scale
        self._shape_for_attn = shape_for_attn
        self._use_query_residual = use_query_residual
        self._qk_channels = qk_channels
        self._v_channels = v_channels

    def __call__(self,
               inputs_q,
               inputs_kv,
               *,
               attention_mask = None,
               is_training):

        dropout_prob = self._dropout_prob if is_training else 0.0
        dropout_attn_prob = self._dropout_attn_prob if is_training else 0.0

        
        output_channels = inputs_q.shape[-1]
        if self._shape_for_attn == 'q':
            qk_channels = inputs_q.shape[-1]
        elif self._shape_for_attn == 'kv':
            qk_channels = inputs_kv.shape[-1]
        else:
            raise ValueError('Unknown value {self._shape_for_attn} for '
                        'shape_for_attention.')

        v_channels = None
        if self._qk_channels is not None:
            qk_channels = self._qk_channels
        if self._v_channels is not None:
            v_channels = self._v_channels

        attention = Attention(
            num_heads=self._num_heads,
            init_scale=self._att_init_scale,
            dropout_prob=dropout_attn_prob,
            qk_channels=qk_channels,
            v_channels=v_channels,
            output_channels=output_channels)(layer_norm(inputs_q),
                                            layer_norm(inputs_kv),
                                            attention_mask=attention_mask)

        attention = hk.dropout(hk.next_rng_key(), dropout_prob, attention)

        # Optionally include a residual to the query.
        # Consider omitting the residual if the semantics of query and output
        # are different, e.g. if queries are positions and outputs are pixels.
        if self._use_query_residual:
            x = bind_list(inputs_q, attention)
        else:
            x = attention

        x = bind_list(x,
                     MLP(widening_factor=self._widening_factor,
                        dropout_prob=dropout_prob,
                        init_scale=self._dense_init_scale)
                        (layer_norm(x), is_training=is_training))
        
        return x