from abc import ABC, abstractmethod
import numpy as np
import torch
import matplotlib.pyplot as plt

class BaseDataVars:
    def __init__(self,
                 path2data = None
                 ):
        self.path2data = path2data

class BaseBatchMaker(ABC):
    def __init__(self,
                 data_vars : BaseDataVars
                 ):
        self.data_vars = data_vars
    def get_sets(self):
        pass

class PfDataVars(BaseDataVars):
    def __init__(self,
                 path2data,
                 data_paths_list=[],
                 epoch_sizes = None,
                 same_trajs_for_all_in_batch = False,
                 ):
        super().__init__(path2data)
        self.nof_steps, self.batch_size, self.nof_batches_per_epoch = epoch_sizes
        self.data_paths_list = data_paths_list
        self.same_trajs_for_all_in_batch = same_trajs_for_all_in_batch

class PfBatchMaker(BaseBatchMaker):
    def __init__(self,
                 data_vars : PfDataVars,
                 sensor_model=None,
                 opt=None
                 ):
        super().__init__(data_vars)
        self.opt = opt
        self.data_vars = data_vars
        self.sm = sensor_model
        self.make_batch_function = self.make_batch_from_trajs_with_sensor_model

    def get_sets(self, nof_sets_to_take=0):
        sets = self.get_gt_trajs_batches_from_files(nof_sets_to_take_for_debug=nof_sets_to_take)
        return sets

    def get_epoch_sets(self,sets):
        _, nof_steps, state_vector_dim = sets.shape
        idcs = np.random.choice(len(sets), self.data_vars.nof_batches_per_epoch * self.data_vars.batch_size, replace=False)
        sets = np.asarray(sets)[idcs].reshape((self.data_vars.nof_batches_per_epoch * self.data_vars.batch_size, 1, nof_steps, state_vector_dim))
        sets = np.transpose(sets, (0, 2, 1, 3))
        return sets

    def get_gt_trajs_batches_from_files(self, nof_sets_to_take_for_debug):
        # prepares the true full trajectory of all targets
        for pf_idx, fp in enumerate(self.data_vars.data_paths_list):
            with open(self.data_vars.path2data+fp, 'rb') as f:
                temp = np.load(f)
                f.close()
                if pf_idx == 0:
                    x_ts = temp
                else:
                    x_ts = np.append(x_ts, temp, axis=0)
        self.data_vars.batch_size = np.maximum(1,self.data_vars.batch_size)
        self.data_vars.batch_size = np.minimum(self.data_vars.batch_size, x_ts.shape[0])
        if self.data_vars.nof_batches_per_epoch != 0:
            self.data_vars.nof_batches_per_epoch = np.minimum(int(x_ts.shape[0]/self.data_vars.batch_size), self.data_vars.nof_batches_per_epoch)
        else:
            self.data_vars.nof_batches_per_epoch = int(x_ts.shape[0]/self.data_vars.batch_size)
        assert self.data_vars.nof_steps <= x_ts.shape[1]
        x_k_arr = np.zeros((self.data_vars.nof_batches_per_epoch, self.data_vars.batch_size, self.data_vars.nof_steps, 1, 4))
        idcs = np.reshape(np.arange(self.data_vars.nof_batches_per_epoch * self.data_vars.batch_size), (self.data_vars.nof_batches_per_epoch, self.data_vars.batch_size, 1))
        for batch_idx in np.arange(self.data_vars.nof_batches_per_epoch):
            for set_idx in np.arange(self.data_vars.batch_size):
                if set_idx == 0 or not self.data_vars.same_trajs_for_all_in_batch:
                    x_k = np.transpose(x_ts[idcs[batch_idx, set_idx], :self.data_vars.nof_steps], (1, 0, 2, 3)).squeeze(-2)
                x_k_arr[batch_idx, set_idx] = x_k
        x_k_arr = np.transpose(x_k_arr, (0,1,3,2,4))
        x_k_arr = np.reshape(x_k_arr,(self.data_vars.nof_batches_per_epoch* self.data_vars.batch_size, self.data_vars.nof_steps, 4))
        return x_k_arr

    def make_batch_from_trajs_with_sensor_model(self, sample_batched, true_sensor_model, device):
        self.data_vars.batch_size = np.maximum(1,self.data_vars.batch_size)
        sample_batched = sample_batched.to(device)
        initial_batch_size, nof_steps, _dc, _dim = sample_batched.shape
        new_batch_size = initial_batch_size
        expected_output2 = torch.reshape(torch.unsqueeze(sample_batched, 1), (new_batch_size, nof_steps, 1, _dim))
        z_k2 = true_sensor_model.get_full_sensor_response_from_prts_locs_torch(expected_output2[:, :, :, (0, 2)])
        for traj_idx in np.arange(self.data_vars.batch_size):
            for set_idx in np.arange(int(self.data_vars.batch_size/self.data_vars.batch_size)):
                idx_in_batch = traj_idx * int(self.data_vars.batch_size / self.data_vars.batch_size) + set_idx
                curr_shape = z_k2[idx_in_batch].shape
                if self.opt.cheat_dont_add_noise_to_meas:
                    z_noise = torch.zeros_like(z_k2[idx_in_batch])
                else:
                    curr_traj_var = np.random.uniform(self.opt.sensor_params.v_var*(self.opt.sensor_params.snr0-self.opt.sensor_params.snr_half_range)/self.opt.sensor_params.snr0,
                                                 self.opt.sensor_params.v_var*(self.opt.sensor_params.snr0+self.opt.sensor_params.snr_half_range)/self.opt.sensor_params.snr0)
                    z_noise = curr_traj_var*torch.randn(curr_shape, device=z_k2.device)
                z_k2[idx_in_batch] += z_noise
        return z_k2, expected_output2


    def paint_z_of_particles(self, particles_z, particles,  set_idcs, time_idcs, sm=None):
        img_rows = np.minimum(len(set_idcs), 2)
        img_cols = np.minimum(len(time_idcs),5)
        assert img_cols <= 8
        assert img_rows <= 6
        curr_interp = 1
        fig, axs = plt.subplots(img_rows, img_cols)
        axs = axs.reshape((img_rows, img_cols))
        plt.suptitle("sensors locations and outputs on "+str(img_cols)+" timesteps of " +str(img_rows)+" trajectories" )
        vmin = np.min(particles_z[set_idcs][:, time_idcs].detach().cpu().numpy())
        vmax = np.max(particles_z[set_idcs][:, time_idcs].detach().cpu().numpy())
        for j in np.arange(img_rows):
            for i in np.arange(img_cols):
                a = axs[j, i].imshow(particles_z[set_idcs[j], time_idcs[i]], extent=[self.opt.sensor_params.center[0] - self.opt.sensor_params.sensor_size[0] / 2 - self.opt.sensor_params.dt / curr_interp / 2,
                                                                                      self.opt.sensor_params.center[0] + self.opt.sensor_params.sensor_size[0] / 2 + self.opt.sensor_params.dt / curr_interp / 2,
                                                                                      self.opt.sensor_params.center[1] - self.opt.sensor_params.sensor_size[1] / 2 - self.opt.sensor_params.dt / curr_interp / 2,
                                                                                      self.opt.sensor_params.center[1] + self.opt.sensor_params.sensor_size[1] / 2 + self.opt.sensor_params.dt / curr_interp / 2], origin='lower', aspect='equal', vmin=vmin, vmax=vmax)
                plt.colorbar(a, orientation='horizontal')
                str1 = "traj_idx="+str(set_idcs[j])+", ts="+str(time_idcs[i])+", loc="
                cnt0 = 0
                for targ in particles[set_idcs[j], time_idcs[i]]:
                    if cnt0 == 4:
                        str1 += '\n'
                    str1 += "[%.1f, %.1f]" % (targ[0], targ[2])
                    cnt0 += 1
                str1 += ""
                axs[j, i].set_title(str1, fontsize=8)
                axs[j, i].grid(color='k', linestyle='-', linewidth=0.5)
                axs[j, i].scatter(particles[set_idcs[j],time_idcs[i]][0, 0], particles[set_idcs[j],time_idcs[i]][0, 2], marker='x', c='r')
                sens_xy = sm.all_z_xy_coo.detach().cpu().numpy()
                for sens_idx in np.arange(sens_xy.shape[0]):
                    axs[j, i].scatter(sens_xy[sens_idx,0], sens_xy[sens_idx,1], marker='o',s=5, c='b')
        plt.show(block=False)
