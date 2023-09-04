import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch
#import pyparticleest as pe
from tempfile import TemporaryFile
#import stonesoup.metricgenerator.ospametric as ospa_metric
#import stonesoup.types.array as ss_tps_arr

import scipy
from scipy.ndimage import gaussian_filter
import time
from MotionModel import MotionModelParams

import torch
from torch.distributions.multivariate_normal import MultivariateNormal  as torch_mulvar_norm
import torch.nn.functional as torch_F
####################################################################
# https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
####################################################################

from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

torch.set_default_dtype(torch.float64)
seed0 = 12
seed0 = 11
seed0 = 16
seed0 = 17
#seed0 = 18

np.random.seed(seed0)
torch.random.manual_seed(seed0)

do_debug_prints = False
#for trjectory painting
colormap = {
    0:'gray',
    1:'r',
    2:'g',
    3:'b',
    4:'c',
    5:'m',
    6:'y',
    7:'k'
}

tau = 1
sig_u = 0.1
sig_u2 = np.power(sig_u,2)
#sig_u2 = 0.1

nof_steps=100
snr0db = 20.0
d0 = 5.0
p=2.0
c=10.0
n=5000
nof_targets=1
nof_targs = nof_targets
nof_parts = 100
nof_steps_for_sim=100

state_vector_dim = 4
time_steps = np.linspace(0,nof_steps*tau,nof_steps+1)
lh_sig2 = 1
#lh_sig2 = 0.8

is_torch_and_not_numpy = True
print("on mode: "+("torch" if is_torch_and_not_numpy else "numpy"))
if is_torch_and_not_numpy:
    device_str = 'cuda'
    device_str = 'cpu'
    if device_str == 'cuda':
        assert torch.cuda.is_available()
    device = torch.device(device_str)
    print("device: "+str(device))

max_parts_to_paint = 100


#model
F = np.kron(np.eye(2), [[1,tau],[0,1]])
Q_tag = [[np.power(tau,3)/3, np.power(tau, 2)/2],[np.power(tau,2)/2, tau]]
Q = sig_u2*np.kron(np.eye(2), Q_tag)

Qs = np.tile(Q,(nof_targets,1,1))
Fs = np.tile(F,(nof_targets,1,1))

center = [100,100]
center = [0,0]
sensor_size = [120,120]
sensor_size = [15*199,5.76*63]

speed_factor = 0.05

# sensor noise
v_var = 1
d0 = 5
dt=10
nof_s_x = int(sensor_size[0]/dt+1)
nof_s_y = int(sensor_size[1]/dt+1)
eps = 1e-18

#configurations
pad_mult=0
interp_sig = 20
interp = 1
do_max_pool = False
# widens the gaussian for the interpolation 0.5 is minimum for highest resoltion
interp_sig_mult = 0.5
#interp_sig_mult = 1
debug_flag = False
print("debug_flag "+str(debug_flag))


print("pad_mult: " + str(pad_mult)+", interp_sig: " + str(interp_sig))

do_interpolate = False
print("do_interpolate "+str(do_interpolate))
if do_interpolate:
    print("interp: "+str(interp))
    print("do_max_pool: "+str(do_max_pool))

    #lh_sig2 = lh_sig2/interp
    #lh_sig2 = 0.5

is_gaussian = False
print("is_gaussian "+str(is_gaussian))

do_threshold_measurements = False
threshold_measurements_th_mult = 0.75
#threshold_measurements_th_mult = 0.5

threshold_measurements_th = snr0db*threshold_measurements_th_mult
print("do_threshold_measurements :"+str(do_threshold_measurements))
if do_threshold_measurements:
    print("threshold_measurements_th_mult :"+str(threshold_measurements_th_mult)+", mult: "+str(threshold_measurements_th_mult))

update_velocities_kalman = True
cheat_get_true_vels = False
cheat_get_true_vels_how_many = 100
print("update_velocities_kalman :"+str(update_velocities_kalman))
if update_velocities_kalman:
    print("cheat_get_true_vels :"+str(cheat_get_true_vels))
    if cheat_get_true_vels:
        print("cheat_get_true_vels_how_many :" + str(cheat_get_true_vels_how_many))


cheat_first_particles = True
print("cheat_first_particles :" + str(cheat_first_particles))
print("lh_sig2: "+str(lh_sig2))

limit_sensor_exp = False
print("limit_sensor_exp :" + str(limit_sensor_exp))
if limit_sensor_exp:
    meas_particle_lh_exp_power_max = 1
    print("meas_particle_lh_exp_power_max :" + str(meas_particle_lh_exp_power_max))
# 1-> minimum paossible match for pixel is 0.3678
# 2-> minimum paossible match for pixel is 0.1353
# 3-> minimum paossible match for pixel is 0.0498
# 4-> minimum paossible match for pixel is 0.0183
# 5-> minimum paossible match for pixel is 0.0067

get_z_for_particles_at_timestep_torch_add_noise = False
print("get_z_for_particles_at_timestep_torch_add_noise :" + str(get_z_for_particles_at_timestep_torch_add_noise))

limit_particles_distance = False
print("limit_particles_distance :" + str(limit_particles_distance))

eliminate_noise = False
print("eliminate_noise :" + str(eliminate_noise))
if eliminate_noise:
    eliminate_noise_lower_th = 2
    print("eliminate_noise_lower_th :" + str(eliminate_noise_lower_th))


cheat_dont_add_noise_to_meas = False
print("cheat_dont_add_noise_to_meas :" + str(cheat_dont_add_noise_to_meas))

update_X_hat_tiled = False
print("update_X_hat_tiled :" + str(update_X_hat_tiled))

mm_params = MotionModelParams(tau=tau, sig_u=sig_u)
is_TRAPP = False
cheat_first_vels = True

class SensorParams:
    def __init__(self,
                 snr0db=20.0,
                 snr0db_max=20.,
                 d0 = 5.,
                 center = [100, 100],
                 sensor_size = [120, 120],
                 v_var = 1,
                 dt = 10,
                 eps = 1e-18
                 ):
        super().__init__()
        self.snr0db = snr0db
        self.snr0db_max = snr0db_max
        self.d0 = d0
        self.center = center
        self.sensor_size = sensor_size
        self.v_var = v_var # sensor noise
        self.dt = dt
        self.eps = eps
        self.nof_s_x = int(sensor_size[0] / dt + 1)
        self.nof_s_y = int(sensor_size[1] / dt + 1)

sensor_params = SensorParams(snr0db=snr0db, d0=d0, center=center, sensor_size=sensor_size, v_var=v_var, dt=dt, eps=eps)

cheat_first_locs_only_half_cheat = False
ospa_p = 2.0
ospa_c = 10.0
sensor_active_dist = 40.0
lh_sig_sqd = 1.0
###############################################################################
# making initial state vector limiting (x,y) positions and velocities accordingly for 1 target
def make_x0(speed_factor=1):
    x0 = np.zeros((state_vector_dim))
    for xy in [0, 1]:
        x0[2*xy] = np.random.uniform(low=center[xy] - sensor_size[xy]/2, high =center[xy] + sensor_size[xy]/2,size=1)
        direction = 1 if x0[2*xy] < center[xy] else -1
        max_speed =  direction*(sensor_size[xy]/2+np.abs(x0[0]-center[xy]))/(tau*nof_steps)
        min_speed = -direction*(sensor_size[xy]/2-np.abs(x0[0]-center[xy]))/(tau*nof_steps)
        x0[2*xy+1] = np.random.uniform(low=min_speed*speed_factor,high=max_speed*speed_factor)
    return x0

# makes 1 target trajectory
def get_target_traj(x0, F, Q, max_iters=1000):
    target_traj = np.zeros((nof_steps+1,state_vector_dim))
    target_traj[0] = x0
    suceeded = False
    trial_idx = 0
    while not suceeded:
        trial_idx += 1
        if trial_idx > max_iters:
            return False
        #print(trial_idx)
        for step_idx in np.arange(nof_steps):
            target_traj[step_idx+1] = np.random.multivariate_normal(np.matmul(F, target_traj[step_idx]).squeeze(), Q, 1).squeeze()
            bad_step = False
            for xy in [0, 1]:
                if target_traj[step_idx, 2 * xy] < center[xy] - sensor_size[xy] / 2 or target_traj[step_idx, 2 * xy] > center[xy] + sensor_size[xy] / 2:
                    bad_step = True
                    break
            if bad_step: break
        if bad_step: continue
        suceeded = True
    return target_traj

#makes nof_targets trjectories using get_target_traj and make_x0
def make_particle_traj(nof_targets):
    x_t = np.zeros((nof_steps + 1, nof_targets, state_vector_dim))
    for target_idx in np.arange(nof_targets):
        x0 = make_x0()
        target_traj = False
        while 1:
            target_traj = get_target_traj(x0, Fs[target_idx], Qs[target_idx])
            if type(target_traj) != bool:
                break
            else:
                print("dfsdfgs")
                x0 = make_x0()
        x_t[:,target_idx] = target_traj
        print("made full traj target_idx: "+str(target_idx))
    return x_t

# makes nof_parts particles with nof_targets tarhgets using make_particle_traj
def make_parts_trajs(nof_parts, nof_targets):
    xs = np.zeros((nof_parts, nof_steps + 1, nof_targets, state_vector_dim))
    for part_idx in np.arange(nof_parts):
        xs[part_idx] = make_particle_traj(nof_targets)
        print("made full particle part_idx: " + str(part_idx))
    return xs

############################################################################
def plot_3d_particle_traj_with_particles(x_t, particles, weights):
    assert len(x_t.shape) == 3
    assert len(particles.shape) == 4
    assert len(weights.shape) == 2
    print(particles.shape[0])
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('time')
    ax.set_xlim(center[0] - sensor_size[0] / 2, center[0] + sensor_size[0] / 2)
    ax.set_ylim(center[1] - sensor_size[1] / 2, center[1] + sensor_size[1] / 2)
    (nof_time_steps, nof_targets, dc1) = x_t.shape
    # time_steps = np.linspace(0, nof_steps * tau, nof_steps + 1)
    time_steps = np.arange(nof_time_steps)
    elev0 = 90
    azim0 = 0
    scatter_size0 = 50

    for target_idx in np.arange(nof_targets):
        # ax.plot3D(x_t[:,target_idx,0], x_t[:, target_idx,2],time_steps, color=colormap[target_idx%len(colormap)], drawstyle='default')
        ax.plot3D(x_t[:, target_idx, 0], x_t[:, target_idx, 2], time_steps, color='gray', drawstyle='default')
        ax.view_init(elev=elev0, azim=azim0)

        #ax.set_zlim(-1, 1);
        ax.scatter3D(x_t[:, target_idx, 0], x_t[:, target_idx, 2], time_steps, c=time_steps, cmap='jet', s=scatter_size0);
        for part in np.arange(particles.shape[1]):
            for time_step in time_steps:
                ax.scatter3D(particles[time_step, part, target_idx, 0], particles[time_step, part, target_idx, 2], time_step, c='gray', cmap='jet', s=10000 * weights[time_step, part]);
    plt.show(block=False)

# makes sensor output from particles, used only in the z of the real targets/state vectors




def get_z_for_particle_in_loop(particles):
    # len(particles.shape) should be 3
    (nof_parts, nof_targets, state_vector_dim) = particles.shape

    z_coo_x = center[0] - sensor_size[0]/2+ np.tile(dt*np.arange(nof_s_x).reshape((1,nof_s_x,1)),[nof_s_y,1,nof_targets])
    z_coo_y = center[1] - sensor_size[1]/2+ np.tile(dt*np.arange(nof_s_y).reshape((nof_s_y,1,1)),[1,nof_s_x,nof_targets])

    nof_parts = particles.shape[0]
    z_snrs = np.zeros((nof_parts, nof_s_y, nof_s_x))
    part_idx = -1
    for curr_part in particles:
        part_idx+=1
        #print(part_idx)
        z_snrs[part_idx] = np.sum(np.minimum(snr0db,
                                             snr0db * d0 * d0 / (eps +
                                                        np.power(z_coo_x - np.tile(curr_part[:, 0].reshape((1, 1, nof_targets)), [nof_s_y, nof_s_x, 1]), 2) +
                                                        np.power(z_coo_y - np.tile(curr_part[:, 2].reshape((1, 1, nof_targets)), [nof_s_y, nof_s_x, 1]), 2)
                                                                 )
                                             ), axis=2)
        #z_snrs[part_idx] = np.sum(snr0db * snr0db / (eps + np.power(z_coo_x - np.tile(curr_part[:, 0].reshape((1, 1, nof_targets)), [nof_s_y, nof_s_x, 1]), 2) + np.power(z_coo_y - np.tile(curr_part[:, 2].reshape((1, 1, nof_targets)), [nof_s_y, nof_s_x, 1]), 2)), axis=2)
    #z_snrs = np.minimum(snr0db, z_snrs)
    return z_snrs

def get_z_for_particle_in_loop_interolate(particles, curr_interp):
    # len(particles.shape) should be 3
    (nof_parts, nof_targets, state_vector_dim) = particles.shape
    nof_s_x_new = int(sensor_size[0] / dt * curr_interp + 1)
    nof_s_y_new = int(sensor_size[1] / dt * curr_interp + 1)
    z_coo_x = np.tile(np.linspace(center[0] - sensor_size[0] / 2, center[0] + sensor_size[0] / 2, num=nof_s_x_new, endpoint=True).reshape((1, nof_s_x_new, 1)), [nof_s_y_new, 1, nof_targets])
    z_coo_y = np.tile(np.linspace(center[1] - sensor_size[1] / 2, center[1] + sensor_size[1] / 2, num=nof_s_y_new, endpoint=True).reshape((nof_s_y_new, 1, 1)), [1, nof_s_x_new, nof_targets])
    nof_parts = particles.shape[0]
    z_snrs = np.zeros((nof_parts, nof_s_y_new, nof_s_x_new))
    part_idx = -1
    for curr_part in particles:
        part_idx+=1
        #print(part_idx)
        z_snrs[part_idx] = np.sum(np.minimum(snr0db,
                                             snr0db * d0 * d0 / (eps +
                                                        np.power(z_coo_x - np.tile(curr_part[:, 0].reshape((1, 1, nof_targets)), [nof_s_y_new, nof_s_x_new, 1]), 2) +
                                                        np.power(z_coo_y - np.tile(curr_part[:, 2].reshape((1, 1, nof_targets)), [nof_s_y_new, nof_s_x_new, 1]), 2)
                                                                 )
                                             ), axis=2)
        #z_snrs[part_idx] = np.sum(snr0db * snr0db / (eps + np.power(z_coo_x - np.tile(curr_part[:, 0].reshape((1, 1, nof_targets)), [nof_s_y, nof_s_x, 1]), 2) + np.power(z_coo_y - np.tile(curr_part[:, 2].reshape((1, 1, nof_targets)), [nof_s_y, nof_s_x, 1]), 2)), axis=2)
    #z_snrs = np.minimum(snr0db, z_snrs)
    return z_snrs





do_run_get_particles = True
if do_run_get_particles:
    file_str = "../pf/particles/pt_parts" + str(nof_parts) + "_tars" + str(nof_targets) + "_steps" + str(nof_steps) + ".npy"
    file_str = "../pf/particles/pt_parts" + str(9000) + "_tars" + str(1) + "_steps" + str(100) + ".npy"
    file_str = "../pf/particles/pt_parts" + str(1000) + "_tars" + str(1) + "_steps" + str(100) + ".npy"


    try:
        with open(file_str, 'rb') as f:
            x_ts = np.load(f)
            f.close()
    except:
        x_ts = make_parts_trajs(nof_parts, nof_targets)
        with open(file_str, 'wb') as f:
            np.save(f, x_ts)





