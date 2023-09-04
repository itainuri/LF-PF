import copy

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch
import pyparticleest as pe
from tempfile import TemporaryFile
import stonesoup.metricgenerator.ospametric as ospa_metric
import stonesoup.types.array as ss_tps_arr

import scipy
from scipy.ndimage import gaussian_filter

import torch
from torch.distributions.multivariate_normal import MultivariateNormal  as torch_mulvar_norm
from torch.distributions.normal import Normal  as torch_normal
import torch.nn.functional as torch_F
from models import *
import torchvision.transforms as torch_transforms
import time as time
import torch
#global cuda_time
#cuda_time = 0
class conv2d_gaussian_filter(torch.nn.Module):
    def __init__(self, kernel_size, sigma):
        super(conv2d_gaussian_filter, self).__init__()
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel*17.157073 * 0.5 / interp_sig_mult * np.sqrt(np.power(sigma * sigma, 2) / (torch.pi * torch.pi))
        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(1, 1, 1, 1)
        self.gaussian_filter_conv2d = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, bias=False, padding=int(kernel_size / 2), padding_mode='zeros')

        self.gaussian_filter_conv2d.weight.data = gaussian_kernel.type(torch.float32)
        self.gaussian_filter_conv2d.weight.requires_grad = False
    def forward(self, input):
        return self.gaussian_filter_conv2d(input)

def get_z_for_particles_at_timestep_torch(particles):
    # particle should be of shape = [nof_particles, nof_targets, state_dim]
    interp_sig = 20
    pad = pad_mult * interp_sig
    pad_tiles = int(np.ceil(pad / dt))
    assert len(particles.shape) == 3
    nof_particles, curr_nof_targets, curr_state_dim = particles.shape
    z_coo_x = center[0] - sensor_size[0] / 2 - pad_tiles * dt + torch.tile(dt * torch.arange(nof_s_x + 2 * pad_tiles, device=device).reshape((1, 1, 1, nof_s_x + 2 * pad_tiles)), [nof_particles, curr_nof_targets, nof_s_y + 2 * pad_tiles, 1])
    z_coo_y = center[1] - sensor_size[1] / 2 - pad_tiles * dt + torch.tile(dt * torch.arange(nof_s_y + 2 * pad_tiles, device=device).reshape((1, 1, nof_s_y + 2 * pad_tiles, 1)), [nof_particles, curr_nof_targets, 1, nof_s_x + 2 * pad_tiles])

    particles_xs = torch.tile(particles[:, :, 0].reshape((*particles.shape[:-1], 1, 1)), [1, 1, nof_s_y + 2 * pad_tiles, nof_s_x + 2 * pad_tiles])
    particles_ys = torch.tile(particles[:, :, 2].reshape((*particles.shape[:-1], 1, 1)), [1, 1, nof_s_y + 2 * pad_tiles, nof_s_x + 2 * pad_tiles])
    z_snrs = torch.sum(torch.minimum(torch.tensor(snr0db), snr0db * d0 * d0 / (eps + torch.pow(z_coo_x - particles_xs, 2) + torch.pow(z_coo_y - particles_ys, 2))), axis=1)
    return z_snrs

def add_gaussian_noise_to_z_for_particles(z_snrs):
    nof_particles, z_y_size, z_y_size = z_snrs.shape

    global gaussian_noise
    try:
        gaussian_noise
    except:
        noise_var = torch.zeros(1, device=device)
        noise_var[0] = v_var
        locs = torch.tile(torch.zeros((1, 1, 1), device=device), (nof_parts, *z_snrs.shape[-2:])).to(device)
        vars = torch.tile(torch.reshape(noise_var, (1, 1, 1)), (nof_parts, *z_snrs.shape[-2:])).to(device)
        gaussian_noise = torch_normal(locs, vars, validate_args=None)

    z_snrs += gaussian_noise.sample()
    # all_z_sum_check = torch.sum(torch.sum(z_snrs, axis=-1), axis=-1)
    #return z_snrs



def interp_z_torch(z_snrs):
    # TODO deletet  z_interp2
    # particle should be of shape = [nof_particles, nof_targets, state_dim]
    assert len(z_snrs.shape) == 3
    global gaussian_conv2d
    global gauss_conv

    curr_nof_parts = z_snrs.shape[0]
    nof_s_x_new = z_snrs.shape[-1]
    nof_s_y_new = z_snrs.shape[-2]
    arange_x_end_interp = interp * nof_s_x_new - ( interp - 1)
    arange_y_end_interp = interp * nof_s_y_new - ( interp - 1)

    sig_y = interp * interp_sig_mult
    sig_x = interp * interp_sig_mult

    #################################################################
    try:
        gaussian_conv2d
    except:
        sigma = interp * interp_sig_mult
        kernel_size_from_sigma = int(np.maximum(5, int(sigma * 6) + (1 if int((sigma * 6) // 2 * 2) == int(sigma * 6) else 0)))
        kernel_size = min(int((arange_y_end_interp // 4) * 2 + 1), kernel_size_from_sigma)
        gaussian_conv2d = conv2d_gaussian_filter(kernel_size, sigma).to(device)
        gauss_conv = torch_transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))

    ################################################################
    z_interp = torch.zeros(curr_nof_parts, arange_y_end_interp, arange_x_end_interp, device=device)
    x_indcs = torch.reshape(torch.tile(interp * torch.unsqueeze(torch.arange(nof_s_x_new), 0), (nof_s_y_new, 1)), (-1,)).to(device)
    y_indcs = torch.reshape(torch.tile(interp * torch.unsqueeze(torch.arange(nof_s_y_new), 1), (1, nof_s_x_new)), (-1,)).to(device)
    z_interp[:, y_indcs, x_indcs] = torch.reshape(z_snrs, (curr_nof_parts, -1))
    if 0:
        #z_interp2 = 17.157073 * 0.5 / interp_sig_mult * np.sqrt(np.power(sig_y * sig_x, 2) / (torch.pi * torch.pi)) * torch.squeeze(gaussian_conv2d(torch.unsqueeze(z_interp, 0)), 0)
        z_interp2 = torch.squeeze(gaussian_conv2d(torch.unsqueeze(z_interp.type(torch.float32), 1)), 1).type(torch.float64)
        if 0:
            fig, axs = plt.subplots(1, 4)
            # plt.sca(axes[1, 1])
            idx = 21
            idx = 0
            axs[0].imshow(z_snrs.cpu().detach().numpy()[idx])
            axs[1].imshow(z_interp.cpu().detach().numpy()[idx])
            axs[2].imshow(z_interp2.cpu().detach().numpy()[idx])
            axs[3].imshow(gaussian_conv2d.gaussian_filter_conv2d.weight.cpu().detach().numpy()[0, 0])
            plt.show(block=False)
    else:
        z_interp2 = 17.157073 * 0.5 / interp_sig_mult * np.sqrt(np.power(sig_y * sig_x, 2) / (torch.pi * torch.pi)) * gauss_conv(z_interp)
        if 0:
            fig, axs = plt.subplots(1, 3)
            # plt.sca(axes[1, 1])
            idx = 21
            idx = 0
            axs[0].imshow(z_snrs.cpu().detach().numpy()[idx])
            axs[1].imshow(z_interp.cpu().detach().numpy()[idx])
            axs[2].imshow(z_interp2.cpu().detach().numpy()[idx])
            plt.show(block=False)
    return z_interp2
###################################################################

def get_lh_measure_particles_with_measurement_to_torch(particles, measurement, device,  return_log = False):
    particles = torch.from_numpy(particles).to(device)
    measurement = torch.from_numpy(measurement).to(device)
    #cuda_time_start = time.time()
    pz_x = get_lh_measure_particles_with_measurement_torch(particles, measurement,  return_log = return_log)
    #cuda_time += time.time() - cuda_time_start
    pz_x = pz_x.cpu().detach().numpy()
    return  pz_x

stride = 1+(interp-1)*do_interpolate
maxpool = torch.nn.MaxPool2d(stride, stride=stride, padding=(int(stride/2),int(stride/2)), dilation=1, return_indices=False, ceil_mode=False).to(device)
avg_conv2d = torch.nn.Conv2d(1, 1, kernel_size=stride, stride=int(stride), padding=(int(stride/2),int(stride/2)), dilation=1, bias=False, padding_mode='zeros', device=device, dtype=None)
avg_conv2d.weight.data = (torch.ones(1,1, stride, stride)/stride/stride).type(torch.float32).to(device)
avg_conv2d.weight.requires_grad = False

def get_lh_measure_particles_with_measurement_torch(particles, measurement, return_log = False):
    # TODO delete z_lh_ln, z_lh_ln2
    assert len(measurement.shape) == 2
    assert len(particles.shape) == 3

    z_for_particles = get_z_for_particles_at_timestep_torch(particles)
    if get_z_for_particles_at_timestep_torch_add_noise:
        add_gaussian_noise_to_z_for_particles(z_for_particles)

    pad_vert = int((z_for_particles.shape[-2]-measurement.shape[-2])/2)
    pad_hor = int((z_for_particles.shape[-1]-measurement.shape[-1])/2)
    measurement = torch_F.pad(measurement,(pad_hor, pad_hor, pad_vert, pad_vert))

    if do_threshold_measurements:
        measurement = torch.minimum(measurement, torch.tensor(threshold_measurements_th).to(device))
        z_for_particles = torch.minimum(z_for_particles, torch.tensor(threshold_measurements_th).to(device))

    if eliminate_noise:
        measurement = torch.max(torch.tensor(eliminate_noise_lower_th, device=device), measurement)
        measurement -= eliminate_noise_lower_th
        z_for_particles = torch.max(torch.tensor(eliminate_noise_lower_th, device=device), z_for_particles)
        z_for_particles-= eliminate_noise_lower_th

    if do_interpolate:
        measurement = torch.squeeze(interp_z_torch(torch.unsqueeze(measurement,0)), 0)
        z_for_particles = interp_z_torch(z_for_particles)

    z_for_meas_rep = torch.tile(measurement, [z_for_particles.shape[0], 1, 1])
    z_lh_ln = -torch.pow(z_for_meas_rep - z_for_particles, 2) / 2 / lh_sig2
    z_lh_ln2 = z_lh_ln.cpu().detach().numpy()# TODO delete

    if do_interpolate and do_max_pool:
        z_lh_ln = maxpool(z_lh_ln)
    #z_lh_ln = torch.squeeze(avg_conv2d(torch.unsqueeze(z_lh_ln.type(torch.float32), 1)), 1).type(torch.float64)

    if limit_sensor_exp and torch.max(z_lh_ln) != torch.min(z_lh_ln):
        z_lh_ln += -torch.max(z_lh_ln)
        indcs = torch.where(z_lh_ln < -meas_particle_lh_exp_power_max)
        z_lh_ln[indcs] += (-z_lh_ln[indcs] - meas_particle_lh_exp_power_max) * (0.5)
        #print(torch.min(z_lh_ln));    print(torch.max(z_lh_ln))
    if 0:
        fig, axs = plt.subplots(1, 4)
        # plt.sca(axes[1, 1])
        idx = 21
        idx = 0
        idx = 74
        # idx = 85
        idx = 12
        fig.suptitle("index: " + str(idx))
        axs[0].imshow(z_for_meas_rep.cpu().detach().numpy()[idx])
        axs[0].set_title("z_for_meas_rep")
        axs[1].imshow(z_for_particles.cpu().detach().numpy()[idx])
        axs[1].set_title("z_for_particles")
        axs[2].imshow(np.exp(z_lh_ln2)[idx])
        axs[2].set_title("np.exp(z_lh_ln2)")
        axs[3].imshow(torch.exp(z_lh_ln).cpu().detach().numpy()[idx])
        axs[3].set_title("torch.exp(z_lh_ln)")
        plt.show(block=False)
    pz_x_log = torch.sum(torch.sum(z_lh_ln, dim=-1), dim=-1)
    if not debug_flag:
        pz_x_log += -torch.max(pz_x_log)
    nof_in_prod = measurement.shape[-2]*measurement.shape[-1]
    if limit_sensor_exp and torch.max(pz_x_log) != torch.min(pz_x_log):
        #print(torch.min(pz_x_log));print(torch.max(pz_x_log))
        exp_eps = 100
        indcs = torch.where(pz_x_log < -exp_eps)
        pz_x_log[indcs] += (-pz_x_log[indcs] - exp_eps) * (1)
        #print(torch.min(pz_x_log))

    if return_log:
        if 0:
            fig, axs = plt.subplots()
            # plt.sca(axes[1, 1])
            axs.imshow(measurement.cpu().detach().numpy())
            plt.show(block=False)
        return pz_x_log
    else:
        return torch.exp(pz_x_log)


###################################################################



def advance_samples_torch(is_mu, old_particles, F, Q, device):
    assert len(old_particles.shape)==3

    global torch_noises
    try:
        torch_noises
    except:

        meas_cov = torch.zeros((state_vector_dim, state_vector_dim), device=device)
        meas_cov[0, 0] = 1
        meas_cov[2, 2] = 1
        meas_cov = 0.1 * meas_cov
        #meas_cov = 0.05* meas_cov
        #meas_cov = 0* meas_cov # 0 is ideal if cheat_grt_true_vels

        locs = torch.tile(torch.zeros((1, 1, old_particles.shape[-1]),device=device), list((*old_particles.shape[:-1], 1)))
        covs = torch.tile(torch.reshape(Q + meas_cov, list((1, 1, *Q.shape))), list((*old_particles.shape[:-1], 1, 1))).to(device)
        torch_noises = torch_mulvar_norm(loc=locs, covariance_matrix=covs)

    if is_mu:
        noises_resampled = torch_noises.mean()
    else:
        noises_resampled = torch_noises.rsample()
    loc_noise_mult = 1
    vel_noise_mult = 1
    #loc_noise_mult = 1
    #vel_noise_mult = 1
    mult_mat = torch.tensor([loc_noise_mult, vel_noise_mult, loc_noise_mult, vel_noise_mult], device=device)
    curr_noise = torch.multiply(noises_resampled[:, :old_particles.shape[1], :], mult_mat)
    #new_particles = torch.transpose(torch.matmul(F, torch.transpose(old_particles, 1, 2)), 1, 2) + 5*noises_resampled[:, :old_particles.shape[1], :]
    new_particles = torch.transpose(torch.matmul(F, torch.transpose(old_particles, 1, 2)), 1, 2) + curr_noise
    return new_particles


def update_atrapp_to_torch(args, device):
    old_particles, old_weights, z_for_meas, is_first_sample_mu, is_TRAPP , F, Q, tvec = args
    old_particles = torch.from_numpy(old_particles).to(device)
    old_weights = torch.from_numpy(old_weights).to(device)
    if z_for_meas is not None:
        z_for_meas = torch.from_numpy(z_for_meas).to(device)
    F = torch.from_numpy(F).to(device)
    Q = torch.from_numpy(Q).to(device)
    args = old_particles, old_weights, z_for_meas, is_first_sample_mu, is_TRAPP , F, Q, tvec
    cuda_time_start = time.time()
    new_particles, final_weights, meas_part_of_initial_sampling, curr_ancestors = update_atrapp_torch(args, device)
    #cuda_time += time.time() - cuda_time_start
    new_particles = new_particles.cpu().detach().numpy()
    final_weights = final_weights.cpu().detach().numpy()
    meas_part_of_initial_sampling = meas_part_of_initial_sampling.cpu().detach().numpy()
    curr_ancestors = curr_ancestors.cpu().detach().numpy()
    return  new_particles, final_weights, meas_part_of_initial_sampling, curr_ancestors

global do_stat
global nof_partents
do_stat = True
nof_partents = np.zeros((nof_steps, nof_targets))

def get_nof_parents():
    if do_stat:
        return nof_partents
    else:
        return None
#update_atrapp_torch0 there is a bug here that makes it work better with debug_flag = True
def update_atrapp_torch0(args, device):
    old_particles, old_weights, z_for_meas, is_first_sample_mu, is_TRAPP , F, Q, tvec = args
    #old_particles.shape = (nof_parts, nof_targs, state_vector_dim)
    measure_ouput_log = True
    time_step_idx = len(tvec)
    assert len(old_particles.shape) == 3
    #print(torch.abs(torch.sum(old_weights)))
    assert torch.abs(torch.sum(old_weights)) <= 1 + 1e-8
    nof_parts, nof_targs, state_vector_dim   = old_particles.shape
    parents_indces = torch.tile(torch.unsqueeze(torch.arange(nof_parts).to(device), 1), (1,nof_targs))
    if z_for_meas is None:
        new_particles = old_particles
        meas_part_of_initial_sampling = torch.ones((nof_parts,),device=device)
        #final_weights = torch.ones((nof_parts,), device=device) / nof_parts
        final_weights =old_weights
    else:
        assert len(z_for_meas.shape) == 2
        if limit_particles_distance:
            included_particles = torch.zeros((nof_parts,), device=device)
        new_particles0 = advance_samples_torch(is_first_sample_mu, old_particles, F, Q, device=device)
        # TODO why different new_particles_for_X_hat?
        new_particles_for_X_hat = new_particles0
        # new_particles_for_X_hat = advance_samples_torch(is_first_sample_mu, old_particles, F, Q, device=device)
        new_particles = torch.zeros_like(old_particles)
        bj_x_kp1 = torch.zeros(old_particles.shape[:-1],device=device)
        weighted_avg_particle = torch.transpose(torch.matmul(torch.transpose(new_particles_for_X_hat, 0, 2), old_weights), 1, 0)
        X_hat = weighted_avg_particle / torch.sum(old_weights)
        #avg_var = torch.pow(new_particles0[:,:,(0,)] -  X_hat[:,(0,)],2) + torch.pow(new_particles0[:,:,(2,)] -  X_hat[:,(2,)],2)
        X_hat_tiled = torch.tile(torch.unsqueeze(X_hat, 0), (nof_parts, 1, 1))
        targets_order = np.random.permutation(nof_targs)
        #targets_order = np.arange(nof_targs)
        for targ_idx in targets_order:

            # print("on target "+str(targ_idx)) #taking same target on all particles
            curr_target_new_particles0 = new_particles0[:, targ_idx].reshape(nof_parts, 1, state_vector_dim)
            # first weighting start
            targs_indcs = (*np.arange(targ_idx), *np.arange(targ_idx + 1, nof_targs))
            Xmj_hat = X_hat_tiled[:, ((targs_indcs))]
            curr_X_hat = torch.cat((Xmj_hat, curr_target_new_particles0), dim=1)
            bj_mu = get_lh_measure_particles_with_measurement_torch(curr_X_hat, z_for_meas, return_log=measure_ouput_log)
            if 0:
                bj_mu = bj_mu/torch.sum(bj_mu)
            new_weights = torch.multiply(old_weights, torch.exp(bj_mu)) if measure_ouput_log else torch.multiply(old_weights, bj_mu)
            ###################################
            if limit_particles_distance:
                radius = 3
                included_particles[:]=0
                included_particles[torch.where(torch.pow(new_particles0[:, targ_idx, 0] - X_hat[targ_idx, 0], 2) +
                                                 torch.pow(new_particles0[:, targ_idx, 2] - X_hat[targ_idx, 2], 2)
                                                 < torch.pow(torch.tensor(radius, device=device), 2)
                                                 )
                                  ] = 1
                # indices = np.where(np.multiply(np.abs(all_parts[time_step_idx, :, target_idx, 0] - wted_avg_traj[time_step_idx, target_idx, 0]) > radius,
                #                               np.abs(all_parts[time_step_idx, :, target_idx, 2] - wted_avg_traj[time_step_idx, target_idx, 2]) > radius))
                new_weights = torch.multiply(included_particles, new_weights)
            #################################
            #if torch.any(torch.isnan(bj_mu),True):
            #    dsfsfs = 4

            new_weights = new_weights / torch.sum(new_weights)    # get_particles_lh_with_real_state(measurement, curr_X_hat, time, sig2)
            # first weighting end (new_weights is lambda on TRAPP alg)
            # resampling
            if 0:
                fig, axs = plt.subplots()
                # plt.sca(axes[1, 1])
                axs.plot(np.sort(new_weights.cpu().detach().numpy()), 'ro')
                plt.show(block=False)
            sampled_indcs_app = torch.multinomial(new_weights, nof_parts, replacement=True)################
            if do_stat: nof_partents[time_step_idx-2, targ_idx] = len(torch.unique(sampled_indcs_app).cpu().detach().numpy())
                #print(sampled_indcs_app)
            # redrawing, advancing-sampling but indices=sampled_indcs_app
            curr_target_old_particles = torch.reshape(old_particles[sampled_indcs_app, targ_idx], (nof_parts, 1, state_vector_dim))
            x_star = advance_samples_torch(False, curr_target_old_particles, F, Q, device=device)
            # bj_x_kp1 is the importance density, or sampling distrbution, but for the specific target on the specific particle
            # bj_x_kp1 dpends on measurement and particle location (disregards weights of any kind)
            if not is_TRAPP:
                curr_target_new_particles = x_star
                bj_x_kp1[:,targ_idx] = bj_mu[sampled_indcs_app]
                parents_indces[:,targ_idx] = sampled_indcs_app
            else:
                # finiding new weights
                curr_X_hat = torch.cat((Xmj_hat, x_star), dim=1)
                bj_x_star = get_lh_measure_particles_with_measurement_torch(curr_X_hat, z_for_meas, return_log=measure_ouput_log)
                rj_x_star =  torch.exp(bj_x_star-bj_mu[sampled_indcs_app]) if measure_ouput_log else  bj_x_star/bj_mu[sampled_indcs_app]
                # normalizing weights
                rj_x_star = rj_x_star/torch.sum(rj_x_star)
                # resampling from x_star, according to rj_x_star
                sampled_indcs_trapp = torch.multinomial(rj_x_star, nof_parts, replacement=True)
                curr_target_new_particles = x_star[sampled_indcs_trapp]
                bj_x_kp1[:, targ_idx] = bj_x_star[sampled_indcs_trapp]
                parents_indces[:,targ_idx] = sampled_indcs_app[sampled_indcs_trapp]

            new_particles[:,targ_idx] = torch.reshape(curr_target_new_particles, (new_particles[:,targ_idx].shape))
            if update_X_hat_tiled:
                X_hat_tiled[:,targ_idx] = torch.matmul(bj_x_kp1[:, targ_idx], new_particles[:,targ_idx])/torch.sum(bj_x_kp1[:, targ_idx]) # for later to use found particles for new particles

        # normalizing bj_x_kp1 so that the multipicazation wont be very small (the pi)
        #

        bj_x_kp1_normed = bj_x_kp1 #/ np.max(bj_x_kp1)
        if 0:
            eps_th = 1e-10
            indcs = torch.where(bj_x_kp1_normed < eps_th)
            bj_x_kp1_normed[indcs] *= torch.pow((eps_th / bj_x_kp1_normed[indcs]), 0.75)
        bj_x_kp1_normed = bj_x_kp1_normed / torch.max(bj_x_kp1_normed)  # / np.max(bj_x_kp1)
        pi_target_bj = torch.prod(bj_x_kp1_normed, dim=1)
        # meausre
        if debug_flag:
            meas_part_of_initial_sampling = pi_target_bj  # / np.max(pi_target_bj)
            final_weights = torch.ones((nof_parts,), device=device) / nof_parts
        else:
            #new_parts_lh = get_lh_measure_particles_with_measurement_torch(new_particles, z_for_meas, return_log=measure_ouput_log)
            #if measure_ouput_log:
            #    log_err = new_parts_lh - pi_target_bj
            #    if torch.max(log_err) != torch.min(log_err):
            #        log_err -= torch.max(log_err)
            #    final_weights = torch.exp(log_err)
            #else:
            #    final_weights = torch.divide(new_parts_lh, pi_target_bj)
            meas_part_of_initial_sampling = torch.ones((nof_parts,),device=device)
    final_weights = final_weights/torch.sum(final_weights)
    # q_of_particle has the effective sampling weight for when the particle was sampled,
    # it depended on the particles level of fitness to the measrements at the time of sampling
    if 0:
        fig, axs = plt.subplots()
        # plt.sca(axes[1, 1])
        axs.imshow(z_for_meas.cpu().detach().numpy())
        plt.show(block=False)
    return new_particles, final_weights, meas_part_of_initial_sampling, parents_indces

#update_atrapp_torch2 suppoed to be best
def update_atrapp_torch2(args, device):
    old_particles, old_weights, z_for_meas, is_first_sample_mu, is_TRAPP , F, Q, tvec = args
    #old_particles.shape = (nof_parts, nof_targs, state_vector_dim)
    measure_ouput_log = True
    normalize_bj_mu_bet_parts = True # gives eaqual weight to each target (not done on alg)
    time_step_idx = len(tvec)
    assert len(old_particles.shape) == 3
    #print(torch.abs(torch.sum(old_weights)))
    assert torch.abs(torch.sum(old_weights)) <= 1 + 1e-8
    nof_parts, nof_targs, state_vector_dim   = old_particles.shape
    ## when here it advances the particles on ts_idx = 1 when y is None, for sime reason its better.
    #new_particles0 = advance_samples_torch(is_first_sample_mu, old_particles, F, Q, device=device)
    ## TODO why different new_particles_for_X_hat?
    #new_particles_for_X_hat = new_particles0
    ##new_particles_for_X_hat = advance_samples_torch(is_first_sample_mu, old_particles, F, Q, device=device)
    parents_indces = torch.tile(torch.unsqueeze(torch.arange(nof_parts).to(device), 1), (1,nof_targs))
    if z_for_meas is None:
        new_particles = old_particles
        meas_part_of_initial_sampling = torch.ones((nof_parts,),device=device)
        #final_weights = torch.ones((nof_parts,), device=device) / nof_parts
        final_weights = old_weights

    else:
        assert len(z_for_meas.shape) == 2
        if limit_particles_distance:
            included_particles = torch.zeros((nof_parts,), device=device)

        new_particles0 = advance_samples_torch(is_first_sample_mu, old_particles, F, Q, device=device)
        # TODO why different new_particles_for_X_hat?
        new_particles_for_X_hat = new_particles0
        # new_particles_for_X_hat = advance_samples_torch(is_first_sample_mu, old_particles, F, Q, device=device)

        new_particles = torch.zeros_like(old_particles)
        bj_x_kp1 = torch.zeros(old_particles.shape[:-1],device=device)
        weighted_avg_particle = torch.transpose(torch.matmul(torch.transpose(new_particles_for_X_hat, 0, 2), old_weights), 1, 0)
        if 0:
            fig, axs = plt.subplots()
            # plt.sca(axes[1, 1])
            axs.imshow(torch.squeeze(get_z_for_particles_at_timestep_torch(torch.unsqueeze(weighted_avg_particle, 0)), 0).cpu().detach().numpy())
            plt.show(block=False)
        X_hat = weighted_avg_particle / torch.sum(old_weights)
        #avg_var = torch.pow(new_particles0[:,:,(0,)] -  X_hat[:,(0,)],2) + torch.pow(new_particles0[:,:,(2,)] -  X_hat[:,(2,)],2)
        X_hat_tiled = torch.tile(torch.unsqueeze(X_hat, 0), (nof_parts, 1, 1))
        targets_order = np.random.permutation(nof_targs)
        #targets_order = np.arange(nof_targs)
        for targ_idx in targets_order:

            # print("on target "+str(targ_idx)) #taking same target on all particles
            curr_target_new_particles0 = new_particles0[:, targ_idx].reshape(nof_parts, 1, state_vector_dim)
            # first weighting start
            targs_indcs = (*np.arange(targ_idx), *np.arange(targ_idx + 1, nof_targs))
            Xmj_hat = X_hat_tiled[:, ((targs_indcs))]
            curr_X_hat = torch.cat((Xmj_hat, curr_target_new_particles0), dim=1)
            bj_mu_log = get_lh_measure_particles_with_measurement_torch(curr_X_hat, z_for_meas, return_log=measure_ouput_log)
            if measure_ouput_log:
                bj_mu_log += -torch.max(bj_mu_log)
                bj_mu = torch.exp(bj_mu_log)
            else:
                bj_mu = bj_mu_log
            if normalize_bj_mu_bet_parts:
                #relevant to: bj_x_kp1[:,targ_idx] = bj_mu[sampled_indcs_app]
                bj_mu = bj_mu/torch.sum(bj_mu)
            new_weights = torch.multiply(old_weights, bj_mu)
            #new_weights = torch.exp(torch.log(old_weights) + bj_mu) if measure_ouput_log else torch.multiply(old_weights, bj_mu)
            ###################################
            if limit_particles_distance:
                radius = 3
                included_particles[:]=0
                included_particles[torch.where(torch.pow(new_particles0[:, targ_idx, 0] - X_hat[targ_idx, 0], 2) +
                                                 torch.pow(new_particles0[:, targ_idx, 2] - X_hat[targ_idx, 2], 2)
                                                 < torch.pow(torch.tensor(radius, device=device), 2)
                                                 )
                                  ] = 1
                # indices = np.where(np.multiply(np.abs(all_parts[time_step_idx, :, target_idx, 0] - wted_avg_traj[time_step_idx, target_idx, 0]) > radius,
                #                               np.abs(all_parts[time_step_idx, :, target_idx, 2] - wted_avg_traj[time_step_idx, target_idx, 2]) > radius))
                new_weights = torch.multiply(included_particles, new_weights)
            #################################
            new_weights = new_weights / torch.sum(new_weights)    # get_particles_lh_with_real_state(measurement, curr_X_hat, time, sig2)
            # first weighting end (new_weights is lambda on TRAPP alg)
            # resampling
            if 0:
                fig, axs = plt.subplots()
                # plt.sca(axes[1, 1])
                axs.plot(np.sort(new_weights.cpu().detach().numpy()), 'ro')
                plt.show(block=False)
            sampled_indcs_app = torch.multinomial(new_weights, nof_parts, replacement=True)################
            if do_stat: nof_partents[time_step_idx-2, targ_idx] = len(torch.unique(sampled_indcs_app).cpu().detach().numpy())
                #print(sampled_indcs_app)
            # redrawing, advancing-sampling but indices=sampled_indcs_app
            curr_target_old_particles = torch.reshape(old_particles[sampled_indcs_app, targ_idx], (nof_parts, 1, state_vector_dim))
            x_star = advance_samples_torch(False, curr_target_old_particles, F, Q, device=device)
            # bj_x_kp1 is the importance density, or sampling distrbution, but for the specific target on the specific particle
            # bj_x_kp1 dpends on measurement and particle location (disregards weights of any kind)
            if not is_TRAPP:
                curr_target_new_particles = x_star
                bj_x_kp1[:,targ_idx] = bj_mu[sampled_indcs_app]
                parents_indces[:,targ_idx] = sampled_indcs_app
            else:
                # finiding new weights
                curr_X_hat = torch.cat((Xmj_hat, x_star), dim=1)
                bj_x_star_log = get_lh_measure_particles_with_measurement_torch(curr_X_hat, z_for_meas, return_log=measure_ouput_log)
                if measure_ouput_log:
                    bj_x_star_log += -torch.max(bj_x_star_log)
                    bj_x_star = torch.exp(bj_x_star_log)
                else:
                    bj_x_star = bj_x_star_log
                rj_x_star =  torch.divide(bj_x_star, bj_mu[sampled_indcs_app])
                # normalizing weights
                rj_x_star = rj_x_star/torch.sum(rj_x_star)
                # resampling from x_star, according to rj_x_star
                sampled_indcs_trapp = torch.multinomial(rj_x_star, nof_parts, replacement=True)
                curr_target_new_particles = x_star[sampled_indcs_trapp]
                bj_x_kp1[:, targ_idx] = bj_x_star[sampled_indcs_trapp]
                parents_indces[:,targ_idx] = sampled_indcs_app[sampled_indcs_trapp]

            new_particles[:,targ_idx] = torch.reshape(curr_target_new_particles, (new_particles[:,targ_idx].shape))
            if update_X_hat_tiled:
                #bj_x_kp1_exp = torch.exp(bj_x_kp1) if measure_ouput_log else bj_x_kp1
                bj_x_kp1_exp = torch.exp(bj_x_kp1[:, targ_idx]) if measure_ouput_log else bj_x_kp1[:, targ_idx]
                X_hat_tiled[:,targ_idx] = torch.matmul(bj_x_kp1_exp[:, targ_idx], new_particles[:,targ_idx])/torch.sum(bj_x_kp1_exp[:, targ_idx]) # for later to use found particles for new particles

        # normalizing bj_x_kp1 so that the multipicazation wont be very small (the pi)
        #

        #bj_x_kp1 = bj_x_kp1 / torch.max(bj_x_kp1)  # / np.max(bj_x_kp1)
        pi_target_bj = torch.prod(bj_x_kp1, dim=1)
        # meausre
        if debug_flag:
            meas_part_of_initial_sampling = pi_target_bj # / np.max(pi_target_bj)
            #meas_part_of_initial_sampling = torch.ones_like(pi_target_bj) if measure_ouput_log else pi_target_bj # / np.max(pi_target_bj)
            final_weights = torch.ones((nof_parts,), device=device) / nof_parts
        else:
            meas_part_of_initial_sampling = torch.ones((nof_parts,),device=device)
            new_parts_lh = get_lh_measure_particles_with_measurement_torch(new_particles, z_for_meas, return_log=measure_ouput_log)

            if measure_ouput_log:
                new_parts_lh += -torch.max(new_parts_lh)
                new_parts_lh = torch.exp(new_parts_lh)
            final_weights = torch.divide(new_parts_lh, pi_target_bj)
        #final_weights = pi_target_bj*pi_target_bj
    #final_weights[:] = 1# = torch.log(final_weights)
    #final_weights = torch.log(torch.e+final_weights)
    final_weights = final_weights/torch.sum(final_weights)
    if 0:
        fig, axs = plt.subplots()
        # plt.sca(axes[1, 1])
        axs.imshow(z_for_meas.cpu().detach().numpy())
        plt.show(block=False)
    if 0:
        fig, axs = plt.subplots()
        # plt.sca(axes[1, 1])
        axs.plot(np.sort(final_weights.cpu().detach().numpy()), 'ro')
        plt.show(block=False)
    # q_of_particle has the effective sampling weight for when the particle was sampled,
    # it depended on the particles level of fitness to the measrements at the time of sampling
    return new_particles, final_weights, meas_part_of_initial_sampling, parents_indces

if debug_flag:
    update_atrapp_torch = update_atrapp_torch0
else:
    update_atrapp_torch = update_atrapp_torch2