import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
        
def current(x, active_inds, t, t_step):
    _, _, full_shape = x
    
    currents = np.zeros(full_shape)
    #access which neurons are active now
    ind = int(t // t_step)
    active = active_inds[ind]
    currents[active] += 1.0
    
    return currents

def define_tgrid(t_span, t_step):
    #calculate the solver grid
    t_start, t_stop = t_span
    n_points = int(np.ceil(t_stop / t_step) + 1) 
    times = np.arange(0, n_points) * t_step
    
    return times

def dz_dt(current_fn, 
            t, 
            z, 
            weight = None, 
            bias = None, 
            leakage: float = -0.2, 
            ang_freq: float = 2 * np.pi,
            arbscale: float = 1.0,):
    """
    Given a function to calculate the current at a moment t and the present
    potential z, calculate the change in potentials
    """
    #the leakage and oscillation parameters are combined to a single complex constant, k
    k = leakage + 1.0j*ang_freq
    
    #multiply current by the weights
    currents = np.matmul(current_fn(t), weight, dtype="complex128")
    #add the bias
    currents += bias

    #update the previous potential and add the currents
    dz = k * z + arbscale * currents

    return dz

def dz_dt_gpu(current_fn, 
            t, 
            z, 
            weight = None, 
            bias = None, 
            leakage: float = -0.2, 
            ang_freq: float = 2 * np.pi,
            arbscale: float = 1.0,):
    """
    Given a function to calculate the current at a moment t and the present
    potential z, calculate the change in potentials. Call GPU for matmul.
    """
    #the leakage and oscillation parameters are combined to a single complex constant, k
    k = leakage + 1.0j*ang_freq
    
    #multiply current by the weights
    currents = jnp.matmul(current_fn(t), weight)
    #add the bias
    currents = currents + bias
    #update the previous potential and add the currents
    dz = k * z + arbscale * currents
    dz = np.array(dz)

    return dz

def find_spikes(sol, threshold: float = 2e-3, sparsity: float = -1.0):
    """
    'Gradient' method for spike detection. Finds where voltages (imaginary component of complex R&F potential) 
    reaches a local minimum & are above a threshold, stores the corresponding time. Sparsity applies a dynamic
    threshold to produce the desired level of spiking. 
    """
    #calculate the temporal extent of the refractory period given its duty cycle
    ts = sol.t
    zs = sol.y
    #find where voltage reaches its max
    voltage = np.imag(zs)
    dvs = np.gradient(voltage, axis=-1)
    dsign = np.sign(dvs)
    #produces ones at the local maxima
    spks = np.diff(dsign, axis=-1, prepend=np.zeros_like((zs.shape[1]))) < 0
    
    #filter by threshold
    above_t = voltage > threshold
    spks = spks * above_t
    
    #last axis of spks_i is time (batch, neuron, time)
    spks_i = np.nonzero(spks)

    #dynamically adjust the threshold to produce sparsity
    if sparsity > 0.0:
        spks_v = np.ravel(voltage[spks_i])
        #find the threshold level necessary to produce the desired sparsity
        dyn_threshold = np.percentile(spks_v, sparsity * 100)
        #filter spikes further by the new threshold
        above_dynt = voltage > dyn_threshold
        spks = spks * above_dynt
        #get the indices of the new spikes
        spks_i = np.nonzero(spks)
        
    #after getting spike time, remove the time index
    spks_t = ts[spks_i[-1]]
    spks_i = spks_i[0:-1]
    #ravel the indices by (batch, neuron)
    shape = spks.shape[0:-1]
    spks_r = [np.ravel_multi_index(spks_i, shape)]

    return (spks_r, spks_t, shape)

def inhibit_midpoint(x, mask_angle: float = 0.0, period: float = 1.0, offset: float = 0.0):
    """
    Given a spike train, remove any spikes occuring within an inhibitory stage
    defined around the center of a period. 
    """
    #pass the method for no inhibitory period
    if mask_angle <= 0.0:
        return x

    inds, times, full_shape = x

    #adjust times by the offset
    adj_times = times + offset
    #take modulo over period
    phases = time_to_phase(adj_times, period)
    #find the phases not within the exclusion angle/inhibitory period
    cond = lambda x: np.abs(x) > mask_angle
    non_inhibited = np.where(cond(phases))

    #remove the inhibited spikes
    inds = [inds[0][non_inhibited]]
    times = times[non_inhibited]

    return (inds, times, full_shape)

def generate_active(x, t_grid, t_box):
    """
    Given the time grid which is being solved over, generate an array which contains the
    active (spiking) neurons at each time step.
    """
    inds, times, full_shape = x
    
    active_inds = []
    
    for (i,t) in enumerate(t_grid):
        cond = lambda x: (x > t - t_box) * (x < t + t_box)
        active = np.nonzero(cond(times))
        #swap the time indices for what flattened neuron they refer to
        active = inds[0][active]

        active = np.unravel_index(active, full_shape)
        active_inds.append(active)
        
    return active_inds

class ODESolution():
    """
    Dummy class to provide right structure of outputs for solutions
    """
    def __init__(self):
        self.t = np.array([])
        self.y = np.array([])

def phase_to_train(x, period: float = 1.0, repeats: int = 3):
    """
    Given a series of input phases defined as a real tensor, convert these values to a 
    temporal spike train: (list of indices, list of firing times, original tensor shape)
    """ 

    t_phase0 = period/2.0
    shape = x.shape
    
    x = x.ravel()

    #list and repeat the index 
    inds = np.nonzero(x)

    #list the time offset for each index and repeat it for repeats cycles
    times = x[inds]

    # order = np.argsort(times)
    
    # #sort by time
    # times = times[order]
    # inds = [np.take(inds[0], order)]
    
    n_t = times.shape[0]
    dtype = x.dtype
    
    times = times * t_phase0 + t_phase0

    #tile across time
    inds = [np.tile(inds[0], (repeats))]
    times = np.tile(times, (repeats))
        
    #create a list of time offsets to move spikes forward by T for repetitions
    offsets = np.arange(0, repeats, dtype=dtype) * period
    offsets = np.repeat(offsets, n_t)

    times += offsets
    
    return (inds, times, shape)

def solve_heun(dz, times, dt, init_val):
    """
    Heun method to provide fine-grained control over solver points and computation
    """

    n_points = len(times)

    #initialize solutions
    y_shape = (*init_val.shape, n_points)
    y = np.zeros(shape=y_shape, dtype=init_val.dtype)
    y[...,0] = init_val
    
    #iterate through
    for (i,t) in enumerate(tqdm(times)):
        #skip solving at the initial condition
        if i == 0:
            continue
        
        #heun method
        slope0 = dz(times[i-1], y[...,i-1])
        y1 = y[...,i-1] + dt*slope0
        slope1 = dz(times[i], y1)

        y[...,i] = y[...,i-1] + dt * (slope0 + slope1) / 2.0

    solution = ODESolution()
    solution.y = y
    solution.t = times

    return solution

def time_to_phase(times, period):
    """
    Given a list of absolute times, use the period to convert them into phases.
    """
    times = times % period
    times = (times - 0.5) * 2.0
    return times

def train_to_phase(spikes, period: float = 1.0, offset: float = 0.0):
    inds, times, full_shape = spikes
    #unravel the indices
    inds = np.unravel_index(inds, full_shape)

    #return array of zeros for no spikes
    if len(times) == 0:
        phases = np.zeros((*full_shape, 1), dtype="float")
        return phases

    #determine the number of cycles in the spike train
    t_max = np.max(times)
    cycles = int(np.ceil(t_max / period)+1)
    
    #make a new copy of times
    times = np.array(times)
    #offset all times according to a global reference
    times += offset
    
    cycle = (times // period).astype("int")
    
    #rescale times into phases
    phase = time_to_phase(times, period)

    full_inds = (*inds, cycle)
    
    phases = np.zeros((*full_shape, cycles), dtype="float")
    phases[full_inds] = phase

    return phases