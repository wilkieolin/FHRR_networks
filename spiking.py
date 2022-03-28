import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

def current(x, t, box = 0.03):
    """
    Given a spike train, calculate the currents at an arbitrary time.
    """
    inds, times, full_shape = x
    
    currents = np.zeros(full_shape)
    
    #get the times in the current window
    cond = lambda x: (x > t - box) * (x < t + box)
    active = np.nonzero(cond(times))
    #swap the time indices for what flattened neuron they refer to
    active = inds[0][active]
    active = np.unravel_index(active, full_shape)
    currents[active] += 1.0
    
    return currents

def dz_dt(current_fn, 
            t, 
            z, 
            weight = None, 
            bias = None, 
            leakage: float = -0.2, 
            ang_freq: float = 2 * np.pi):
    """
    Given a function to calculate the current at a moment t and the present
    potential z, calculate the change in potentials
    """
    #the leakage and oscillation parameters are combined to a single complex constant, k
    k = leakage + 1.0j*ang_freq
    
    #multiply current by the weights
    currents = np.matmul(current_fn(t), weight, dtype="complex")
    #add the bias
    currents += bias

    #update the previous potential and add the currents
    dz = k * z + currents
    return dz

def dz_dt_gpu(current_fn, 
            t, 
            z, 
            weight = None, 
            bias = None, 
            leakage: float = -0.2, 
            ang_freq: float = 2 * np.pi):
    """
    Given a function to calculate the current at a moment t and the present
    potential z, calculate the change in potentials. Call GPU for matmul.
    """
    #the leakage and oscillation parameters are combined to a single complex constant, k
    k = leakage + 1.0j*ang_freq
    
    #multiply current by the weights
    currents = jnp.matmul(current_fn(t), weight)
    currents = np.array(currents, dtype="complex")
    #add the bias
    currents += bias

    #update the previous potential and add the currents
    dz = k * z + currents
    return dz

def findspks(sol, threshold=2e-3):
    """
    'Gradient' method for spike detection. Finds where voltages (imaginary component of complex R&F potential) 
    reaches a local minimum & are above a threshold, stores the corresponding time. 
    """
    #calculate the temporal extent of the refractory period given its duty cycle
    ts = sol.t
    zs = sol.y
    #find where voltage reaches its max
    voltage = np.imag(zs)
    dvs = np.gradient(voltage, axis=-1)
    dsign = np.sign(dvs)
    spks = np.diff(dsign, axis=-1, prepend=np.zeros_like((zs.shape[1]))) < 0
    
    #filter by threshold
    above_t = voltage > threshold
    spks = spks * above_t
    
    #last axis of spks_i is time
    spks_i = np.nonzero(spks)
    #after getting spike time, remove the time index
    spks_t = ts[spks_i[-1]]
    spks_i = spks_i[0:-1]
    #ravel the indices
    shape = spks.shape[0:-1]
    spks_r = [np.ravel_multi_index(spks_i, shape)]

    return (spks_r, spks_t, shape)

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

def solve_heun(dx, tspan, init_val, dt):
    """
    Heun method to provide fine-grained control over solver points and computation
    """
    
    #calculate the solver grid
    t_start, t_stop = tspan
    n_points = int(np.ceil(t_stop / dt) + 1) 
    times = np.arange(0, n_points) * dt

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
        slope0 = dx(times[i-1], y[...,i-1])
        y1 = y[...,i-1] + dt*slope0
        slope1 = dx(times[i], y1)

        y[...,i] = y[...,i-1] + dt * (slope0 + slope1) / 2.0

    solution = ODESolution()
    solution.y = y
    solution.t = times

    return solution

def train_to_phase(spikes, period: float = 1.0, offset: float = 0.0):
    inds, times, full_shape = spikes
    #unravel the indices
    inds = np.unravel_index(inds, full_shape)
    t_max = np.max(times)
    t_phase0 = period / 2.0
    #determine the number of cycles in the spike train
    cycles = int(np.ceil(t_max / period)+1)
    
    #offset all times according to a global reference
    times += offset 
    
    cycle = (times // period).astype("int")
    
    times = (times - t_phase0) / t_phase0
    times = (times + 1.0) % 2.0 - 1.0

    full_inds = (*inds, cycle)
    
    phases = np.zeros((*full_shape, cycles), dtype="float")
    phases[full_inds] = times

    return phases