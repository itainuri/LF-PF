import torch
import numpy as np
import torch.nn.functional as torch_F
from torch.distributions.normal import Normal  as torch_normal

class SensorModel(object):
    def __init__(self, opt, device):
        self.device = device
        self.opt = opt
        #self.reset(opt, device)
    def reset(self, opt, x, device):
        self.opt = opt # TODO check if should be self
        self.get_lh_measure_prts_locs_with_measurement_torch = self.get_lh_measure_prts_locs_with_measurement_torch4
        self.get_active_sensors_mask_from_old_parts = self.get_active_sensors_mask_from_old_parts_new
        self.z_xy_coo = self.get_z_coo_xy_for_all_sensors2(opt, device)


    def get_z_coo_xy_for_all_sensors2(self, opt, device):
        # particle should be of shape = [batch_size, nof_particles, nof_targets, state_dim]
        interp_sig = 20
        pad_mult = 0#1
        loc_vector_dim = 2
        pad = pad_mult * interp_sig
        dt = opt.sensor_params.dt
        sensor_size = opt.sensor_params.sensor_size
        nof_s_x = opt.sensor_params.nof_s_x
        nof_s_y = opt.sensor_params.nof_s_y
        center = opt.sensor_params.center
        snr0db = opt.sensor_params.snr0db
        d0 = opt.sensor_params.d0
        eps = opt.sensor_params.eps
        pad_tiles = int(np.ceil(pad / dt))
        z_coo_x = center[0] - sensor_size[0] / 2 - pad_tiles * dt + torch.tile(dt * torch.arange(nof_s_x + 2 * pad_tiles, device=device).reshape(( 1, nof_s_x + 2 * pad_tiles)), [nof_s_y + 2 * pad_tiles, 1])
        z_coo_y = center[1] - sensor_size[1] / 2 - pad_tiles * dt + torch.tile(dt * torch.arange(nof_s_y + 2 * pad_tiles, device=device).reshape(( nof_s_y + 2 * pad_tiles, 1)), [1, nof_s_x + 2 * pad_tiles])
        z_coo_xy = torch.cat((torch.unsqueeze(z_coo_x,-1),torch.unsqueeze(z_coo_y,-1)), dim=-1)
        z_coo_xy = torch.reshape(z_coo_xy,(nof_s_x*nof_s_y,loc_vector_dim))

        return z_coo_xy


    def get_z_for_prts_locs_at_timestep_torch3(self, opt, prts_locs, z_coo_xy, device):
        # particle should be of shape = [batch_size, nof_particles, nof_targets, state_dim]
        interp_sig = 20
        #pad_mult = 0#1
        #pad = pad_mult * interp_sig
        #dt = opt.sensor_params.dt
        #sensor_size = opt.sensor_params.sensor_size
        #nof_s_x = opt.sensor_params.nof_s_x
        #nof_s_y = opt.sensor_params.nof_s_y
        state_vector_dim=2
        #center = opt.sensor_params.center
        snr0db = opt.sensor_params.snr0db

        d0 = opt.sensor_params.d0
        eps = opt.sensor_params.eps
        #pad_tiles = int(np.ceil(pad / dt))
        #assert len(prts_locs.shape) == 4
        #batch_size, nof_particles, curr_nof_targets, _ = prts_locs.shape
        nof_active_sensors = z_coo_xy.shape[2]
        #particles_xs = torch.tile(particles[:, :, :, 0].reshape((*particles.shape[:-1], 1, 1)), [1, 1, 1, nof_s_y + 2 * pad_tiles, nof_s_x + 2 * pad_tiles])
        #particles_ys = torch.tile(particles[:, :, :, 2].reshape((*particles.shape[:-1], 1, 1)), [1, 1, 1, nof_s_y + 2 * pad_tiles, nof_s_x + 2 * pad_tiles])
        particles_xys = torch.tile(prts_locs.reshape((*prts_locs.shape[:-1], 1, prts_locs.shape[-1])), [1, 1, nof_active_sensors, 1])
        temp0 = snr0db * d0 * d0 / torch.sum(eps + torch.pow(z_coo_xy - particles_xys, 2), dim=-1)
        #temp0 = snr0db * d0 * d0 / (eps + torch.pow(z_coo_x - particles_xs, 2) + torch.pow(z_coo_y - particles_ys, 2))
        temp = torch.where(temp0 <= snr0db, temp0, snr0db)
        z_snrs = torch.sum(temp, axis=1)
        return z_snrs


    def add_gaussian_noise_to_z_for_particles(self, opt, z_snrs, device):
        nof_particles, z_y_size, z_y_size = z_snrs.shape
        nof_parts = opt.nof_parts
        global gaussian_noise
        try:
            gaussian_noise
        except:
            noise_var = torch.zeros(1, device=device)
            noise_var[0] = opt.v_var
            locs = torch.tile(torch.zeros((1, 1, 1), device=device), (nof_parts, *z_snrs.shape[-2:])).to(device)
            vars = torch.tile(torch.reshape(noise_var, (1, 1, 1)), (nof_parts, *z_snrs.shape[-2:])).to(device)
            gaussian_noise = torch_normal(locs, vars, validate_args=None)

        z_snrs = z_snrs+gaussian_noise.sample()
        # all_z_sum_check = torch.sum(torch.sum(z_snrs, axis=-1), axis=-1)
        #return z_snrs


    def get_lh_measure_prts_locs_with_measurement_torch4(self, opt, prts_locs, measurement, active_sensors_mask = None, return_log = False, device=None):
        # TODO delete z_lh_ln, z_lh_ln2
        get_z_for_particles_at_timestep_torch_add_noise = False
        limit_sensor_exp = False
        meas_particle_lh_exp_power_max = 1000
        assert len(measurement.shape) == 3
        assert len(prts_locs.shape) == 4
        loc_vector_dim=2
        batch_size, nof_particles, curr_nof_targets, _ = prts_locs.shape
        active_sensors_idcs = torch.nonzero(torch.reshape(active_sensors_mask,(batch_size, measurement.shape[-2]*measurement.shape[-1])), as_tuple=False)
        first_idx_of_batch_idx = torch.searchsorted(active_sensors_idcs[:, 0], torch.arange(batch_size, device=device), side='right')
        measurement_flat = torch.reshape(measurement, (batch_size, measurement.shape[-2]*measurement.shape[-1]))
        measurement_flat3 = measurement_flat[active_sensors_idcs[:, 0], active_sensors_idcs[:, 1]]
        z_xy_coo3 = self.z_xy_coo[active_sensors_idcs[:, 1]]

        snr0db = opt.sensor_params.snr0db
        d0 = opt.sensor_params.d0
        eps = opt.sensor_params.eps
        particles_xys = prts_locs[active_sensors_idcs[:, 0]]
        particles_xys = prts_locs[active_sensors_idcs[:, 0], active_sensors_idcs[:, 1]]
        temp0 = snr0db * d0 * d0 / torch.sum(eps + torch.pow(torch.reshape(z_xy_coo3, (z_xy_coo3.shape[0],1,1,z_xy_coo3.shape[1])) - particles_xys, 2), dim=-1)
        parts_sensor_response_flat = torch.sum(torch.where(temp0 <= snr0db, temp0, snr0db), axis=2)
        #parts_sensor_response_flat = -0.5 * torch.pow((parts_sensor_response_flat - torch.unsqueeze(measurement_flat3,-1)) / opt.lh_sig_sqd, 2) / opt.lh_sig_sqd
        parts_sensor_response_flat = torch.multiply(
            torch.exp(
                -torch.pow(parts_sensor_response_flat,2)),
            torch.special.i0(2*torch.multiply(parts_sensor_response_flat, torch.unsqueeze(measurement_flat3,-1)),out=None))

        pz_x_log = torch.zeros((batch_size,nof_particles), device=device)
        first_idx = 0
        for idx_in_batch in np.arange(first_idx_of_batch_idx.shape[0]):
            pz_x_log[idx_in_batch] = torch.sum(parts_sensor_response_flat[first_idx:first_idx_of_batch_idx[idx_in_batch]], dim=0)
            first_idx = first_idx_of_batch_idx[idx_in_batch]
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
        if return_log:
            if 0:
                fig, axs = plt.subplots()
                # plt.sca(axes[1, 1])
                axs.imshow(measurement.cpu().detach().numpy())
                plt.show(block=False)
            return pz_x_log
        else:
            return torch.exp(pz_x_log)




    def get_lh_measure_prts_locs_with_measurement_torch3(self, opt, prts_locs, measurement, active_sensors_mask = None, return_log = False, device=None):
        # TODO delete z_lh_ln, z_lh_ln2
        get_z_for_particles_at_timestep_torch_add_noise = False
        limit_sensor_exp = False
        meas_particle_lh_exp_power_max = 1000
        assert len(measurement.shape) == 3
        assert len(prts_locs.shape) == 4
        loc_vector_dim=2
        batch_size, nof_particles, curr_nof_targets, _ = prts_locs.shape
        active_sensors_idcs = torch.nonzero(torch.reshape(active_sensors_mask,(batch_size, measurement.shape[-2]*measurement.shape[-1])), as_tuple=False)
        first_idx_of_batch_idx = torch.searchsorted(active_sensors_idcs[:, 0], torch.arange(batch_size, device=device), side='right')
        measurement_flat = torch.reshape(measurement, (batch_size, measurement.shape[-2]*measurement.shape[-1]))
        measurement_flat3 = measurement_flat[active_sensors_idcs[:, 0], active_sensors_idcs[:, 1]]
        z_xy_coo3 = self.z_xy_coo[active_sensors_idcs[:, 1]]
        snr0db = opt.sensor_params.snr0db
        d0 = opt.sensor_params.d0
        eps = opt.sensor_params.eps
        particles_xys = prts_locs[active_sensors_idcs[:, 0]]
        #particles_xys = prts_locs[active_sensors_idcs[:, 0], active_sensors_idcs[:, 1]]
        temp0 = snr0db * d0 * d0 / torch.sum(eps + torch.pow(torch.reshape(z_xy_coo3, (z_xy_coo3.shape[0],1,1,z_xy_coo3.shape[1])) - particles_xys, 2), dim=-1)
        parts_sensor_response_flat = torch.sum(torch.where(temp0 <= snr0db, temp0, snr0db), axis=2)
        parts_sensor_response_flat = -0.5 * torch.pow((parts_sensor_response_flat - torch.unsqueeze(measurement_flat3,-1)) / opt.lh_sig_sqd, 2) / opt.lh_sig_sqd
        pz_x_log = torch.zeros((batch_size,nof_particles), device=device)
        first_idx = 0
        for idx_in_batch in np.arange(first_idx_of_batch_idx.shape[0]):
            pz_x_log[idx_in_batch] = torch.sum(parts_sensor_response_flat[first_idx:first_idx_of_batch_idx[idx_in_batch]], dim=0)
            first_idx = first_idx_of_batch_idx[idx_in_batch]
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
        if return_log:
            if 0:
                fig, axs = plt.subplots()
                # plt.sca(axes[1, 1])
                axs.imshow(measurement.cpu().detach().numpy())
                plt.show(block=False)
            return pz_x_log
        else:
            return torch.exp(pz_x_log)


    def get_active_sensors_mask_from_old_parts_new(self, opt, mm, prts_locs, prts_vels, ln_weights, device):
        #up to 08.07.2023  there was a bug : used weighted_avg_loc instead of new_weighted_avg_loc  at get_active_sensors_mask_from_old_parts_old
        batch_size, nof_parts, nof_targs, state_loc_vector_dim = prts_locs.shape
        weights = torch.softmax(ln_weights.detach(), dim=1)
        weighted_avg_loc = torch.bmm(weights.view(batch_size, 1, nof_parts), prts_locs.detach().reshape(batch_size, nof_parts, -1)).view(batch_size,1, nof_targs, state_loc_vector_dim)
        weighted_avg_vel = torch.bmm(weights.view(batch_size, 1, nof_parts), prts_vels.detach().reshape(batch_size, nof_parts, -1)).view(batch_size,1, nof_targs, state_loc_vector_dim)
        new_weighted_avg_loc, new_weighted_avg_vel = mm.advance_locations(opt, True, weighted_avg_loc, weighted_avg_vel, device, print_seed=False, print_grad=True)
        dt = opt.sensor_params.dt
        sensor_size = opt.sensor_params.sensor_size
        nof_s_x = opt.sensor_params.nof_s_x
        nof_s_y = opt.sensor_params.nof_s_y
        center = opt.sensor_params.center
        snr0db = opt.sensor_params.snr0db
        d0 = opt.sensor_params.d0
        eps = opt.sensor_params.eps
        pad_tiles = 0

        assert len(prts_locs.shape) == 4
        batch_size, nof_particles, curr_nof_targets, _ = new_weighted_avg_loc.shape
        z_coo_x = center[0] - sensor_size[0] / 2 - pad_tiles * dt + torch.tile(dt * torch.arange(nof_s_x + 2 * pad_tiles, device=device).reshape((1, 1, 1, 1, nof_s_x + 2 * pad_tiles)), [batch_size, nof_particles, curr_nof_targets, nof_s_y + 2 * pad_tiles, 1])
        z_coo_y = center[1] - sensor_size[1] / 2 - pad_tiles * dt + torch.tile(dt * torch.arange(nof_s_y + 2 * pad_tiles, device=device).reshape((1, 1, 1, nof_s_y + 2 * pad_tiles, 1)), [batch_size, nof_particles, curr_nof_targets, 1, nof_s_x + 2 * pad_tiles])
        z_coo_xy = torch.cat((torch.unsqueeze(z_coo_x, -1), torch.unsqueeze(z_coo_y, -1)), dim=-1)
        particles_xys = torch.tile(new_weighted_avg_loc.reshape((*new_weighted_avg_loc.shape[:-1], 1, 1, new_weighted_avg_loc.shape[-1])), [1, 1, 1, nof_s_y + 2 * pad_tiles, nof_s_x + 2 * pad_tiles, 1])
        per_sensor_per_target_dist = torch.sqrt(torch.sum(torch.pow(z_coo_xy - particles_xys, 2), dim=-1))
        per_batch_active_sensors = torch.any(torch.where(per_sensor_per_target_dist <= opt.sensor_active_dist, True, False), 2)
        return torch.squeeze(per_batch_active_sensors, 1)
