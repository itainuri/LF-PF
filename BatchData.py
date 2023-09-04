import copy

import numpy as np
import torch
import matplotlib.pyplot as plt

class BatchData(object):
    def __init__(self, batch_size, nof_steps, nof_parts, nof_targs, device):
        self.device = device
        self.nof_parts = nof_parts
        self.reset(batch_size, nof_steps, nof_parts, nof_targs, device)
    def reset(self, batch_size, nof_steps, nof_parts, nof_targs, device):
        #curr_batch_size, curr_nof_steps, nof_targs, _ = x_with_t0.shape

        # first particles and weights (time 0) are created by: self.model.create_initial_estimate(x0, self.opt.nof_parts)
        # following particles depend on inputs as well (inputs is None for time 0, and not None for time = 1,2,..)
        self.prts_locs_per_iter = torch.zeros((batch_size, nof_steps, self.nof_parts,nof_targs, 2), device=device)
        self.prts_vels_per_iter = torch.zeros((batch_size, nof_steps, self.nof_parts,nof_targs, 2), device=device)
        self.weights_per_iter = torch.zeros((batch_size, nof_steps , self.nof_parts), device=device)
        self.lnw_per_iter = torch.zeros((batch_size, nof_steps , self.nof_parts), device=device)
        self.parents_incs_per_iter = torch.zeros((batch_size, nof_steps , self.nof_parts,nof_targs), device=device)
        self.nn1_lnw_in_per_iter = torch.zeros((batch_size, nof_steps , self.nof_parts,nof_targs), device=device)
        self.nn1_lnw_out_per_iter = torch.zeros((batch_size, nof_steps , self.nof_parts,nof_targs), device=device)
        self.nn3_in_full_parts_weights_var = torch.zeros((batch_size, nof_steps ), device=device)
        self.nn3_in_full_parts_weights = torch.zeros((batch_size, nof_steps , self.nof_parts), device=device)
        self.nn3_in_full_parts_lnw = torch.zeros((batch_size, nof_steps , self.nof_parts), device=device)
        self.nn3_in_unscaled_parts_locs = torch.zeros((batch_size, nof_steps , self.nof_parts,nof_targs, 2), device=device)

    def not_require_grad_all(self):
        self.prts_locs_per_iter.requires_grad = False
        self.prts_vels_per_iter.requires_grad = False
        self.weights_per_iter.requires_grad = False
        self.lnw_per_iter.requires_grad = False
        self.parents_incs_per_iter.requires_grad = False
        self.nn1_lnw_in_per_iter.requires_grad = False
        self.nn1_lnw_out_per_iter.requires_grad = False
        self.nn3_in_full_parts_weights_var.requires_grad = False
        self.nn3_in_full_parts_weights.requires_grad = False
        self.nn3_in_full_parts_lnw.requires_grad = False
        self.nn3_in_unscaled_parts_locs.requires_grad = False

    def detach_all(self):
        self.prts_locs_per_iter = self.prts_locs_per_iter.detach()
        self.prts_vels_per_iter = self.prts_vels_per_iter.detach()
        self.weights_per_iter = self.weights_per_iter.detach()
        self.lnw_per_iter = self.lnw_per_iter.detach()
        self.parents_incs_per_iter = self.parents_incs_per_iter.detach()
        self.nn1_lnw_in_per_iter = self.nn1_lnw_in_per_iter.detach()
        self.nn1_lnw_out_per_iter = self.nn1_lnw_out_per_iter.detach()
        self.nn3_in_full_parts_weights_var = self.nn3_in_full_parts_weights_var.detach()
        self.nn3_in_full_parts_weights = self.nn3_in_full_parts_weights.detach()
        self.nn3_in_full_parts_lnw = self.nn3_in_full_parts_lnw.detach()
        self.nn3_in_unscaled_parts_locs = self.nn3_in_unscaled_parts_locs.detach()

    def make_clean_new_bd(self):
        batch_size, nof_steps, nof_parts, nof_targs, _ = self.prts_locs_per_iter.shape
        new_bd = BatchData(batch_size, nof_steps, nof_parts, nof_targs, self.device)
        return new_bd

    def return_detached_clone(self):
        batch_size, nof_steps, nof_parts, nof_targs, _ = self.prts_locs_per_iter.shape
        new_bd = BatchData(batch_size, nof_steps, nof_parts, nof_targs, self.device)
        new_bd.prts_locs_per_iter = torch.clone(self.prts_locs_per_iter.detach())
        new_bd.prts_vels_per_iter = torch.clone(self.prts_vels_per_iter.detach())
        new_bd.weights_per_iter = torch.clone(self.weights_per_iter.detach())
        new_bd.lnw_per_iter = torch.clone(self.lnw_per_iter.detach())
        new_bd.parents_incs_per_iter = torch.clone(self.parents_incs_per_iter.detach())
        new_bd.nn1_lnw_in_per_iter = torch.clone(self.nn1_lnw_in_per_iter.detach())
        new_bd.nn1_lnw_out_per_iter = torch.clone(self.nn1_lnw_out_per_iter.detach())
        new_bd.nn3_in_full_parts_weights_var = torch.clone(self.nn3_in_full_parts_weights_var.detach())
        new_bd.nn3_in_full_parts_weights = torch.clone(self.nn3_in_full_parts_weights.detach())
        new_bd.nn3_in_full_parts_lnw = torch.clone(self.nn3_in_full_parts_lnw.detach())
        new_bd.nn3_in_unscaled_parts_locs = torch.clone(self.nn3_in_unscaled_parts_locs.detach())
        return new_bd

    def move_to_cpu(self):
        self.device = 'cpu'
        self.prts_locs_per_iter = self.prts_locs_per_iter.detach().cpu()
        self.prts_vels_per_iter = self.prts_vels_per_iter.detach().cpu()
        self.weights_per_iter = self.weights_per_iter.detach().cpu()
        self.lnw_per_iter = self.lnw_per_iter.detach().cpu()
        self.parents_incs_per_iter = self.parents_incs_per_iter.detach().cpu()
        self.nn1_lnw_in_per_iter = self.nn1_lnw_in_per_iter.detach().cpu()
        self.nn1_lnw_out_per_iter = self.nn1_lnw_out_per_iter.detach().cpu()
        self.nn3_in_full_parts_weights_var = self.nn3_in_full_parts_weights_var.detach().cpu()
        self.nn3_in_full_parts_weights = self.nn3_in_full_parts_weights.detach().cpu()
        self.nn3_in_full_parts_lnw = self.nn3_in_full_parts_lnw.detach().cpu()
        self.nn3_in_unscaled_parts_locs = self.nn3_in_unscaled_parts_locs.detach().cpu()
        return self

    def get_ass2true_map(self, x_with_t0, wted_avg_traj, p=2, c=10):
        x_wt0_locs = x_with_t0[:, :, :, [0, 2]]
        batch_size, nof_times, nof_targs, loc_dim = x_wt0_locs.shape
        ass2true_map = -np.ones((batch_size, nof_times, nof_targs))
        #print(x_wt0_locs)
        wted_avg_locs = wted_avg_traj[:, :, :, [0, 2]]
        #print(wted_avg_locs)
        big_number = 1e+16
        batch_set_idx = 0
        for batch_set_idx in np.arange(batch_size):
            for time_step_idx in np.arange(nof_times):
                x_set = np.expand_dims(x_wt0_locs[batch_set_idx, time_step_idx], 0)
                avg_set = np.expand_dims(wted_avg_locs[batch_set_idx, time_step_idx], 1)
                dists = np.sum(np.square(x_set - avg_set), axis=2)
                for targ_counter in np.arange(nof_targs):
                    avg_idx, x_idx = np.unravel_index(np.argmin(dists), dists.shape)
                    ass2true_map[batch_set_idx,time_step_idx, avg_idx] = x_idx
                    dists[avg_idx] = big_number
                    dists[:, x_idx] = big_number
                    #print(ass2true_map[idx])
        return ass2true_map

    def get_parts_wted_var(self, is_nn3_in = False):
        batch_size, nof_steps, _, nof_targs, __ = self.prts_locs_per_iter.shape
        state_vector_dim = 4
        nof_parts = self.weights_per_iter.shape[-1]
        loc_vector_dim  = 2
        if not is_nn3_in:
            parts_locs = self.prts_locs_per_iter
            parts_wts = self.weights_per_iter
        else:
            parts_locs = self.nn3_in_unscaled_parts_locs
            parts_wts = torch.softmax(self.nn3_in_full_parts_lnw, dim=2)
        ff_locs = parts_locs.reshape(batch_size * (nof_steps), nof_parts, loc_vector_dim * nof_targs)
        gg_wts = parts_wts.view(batch_size * (nof_steps), 1, nof_parts)

        all_ts_avg_prts_locs = torch.squeeze(torch.bmm(gg_wts, ff_locs)).view(batch_size, (nof_steps), nof_targs, loc_vector_dim)
        dists_sqd = torch.pow(parts_locs - torch.unsqueeze(all_ts_avg_prts_locs, 2), 2)
        ff_dists = dists_sqd.reshape(batch_size * (nof_steps), nof_parts, loc_vector_dim * nof_targs)
        #all_ts_avg_sqd_dists = torch.squeeze(torch.bmm(gg_wts, ff_dists)).view(batch_size, (nof_steps), nof_targs, loc_vector_dim)
        all_ts_avg_sqd_dists = torch.reshape(torch.squeeze(torch.bmm(gg_wts, ff_dists)),(batch_size, (nof_steps), nof_targs, loc_vector_dim))
        return all_ts_avg_sqd_dists

    def get_nn3_parts_unwted_var(self, is_nn3_in = False):
        if not is_nn3_in:
            parts_locs = self.prts_locs_per_iter
        else:
            parts_locs = self.nn3_in_unscaled_parts_locs
        all_ts_avg_sqd_dists = torch.var(parts_locs,dim=2)
        return all_ts_avg_sqd_dists

    def get_all_ts_avg_particles_and_mapping(self, x_with_t0 = None):
        if x_with_t0 is not None:
            batch_size, nof_steps, nof_targs, state_vector_dim = x_with_t0.shape
        else:
            batch_size, nof_steps, _, nof_targs, __ = self.prts_locs_per_iter.shape
            state_vector_dim = 4
            nof_steps = nof_steps -0*1
        assert nof_steps == self.prts_locs_per_iter.shape[1]-0*1
        nof_parts = self.weights_per_iter.shape[-1]
        loc_vector_dim  = 2
        ff_locs = self.prts_locs_per_iter.reshape(batch_size * (nof_steps + 0*1), nof_parts, loc_vector_dim * nof_targs)
        ff_vels = self.prts_vels_per_iter.reshape(batch_size * (nof_steps + 0*1), nof_parts, loc_vector_dim * nof_targs)
        gg = self.weights_per_iter.view(batch_size * (nof_steps + 0*1), 1, nof_parts)
        all_ts_avg_prts_locs = torch.squeeze(torch.bmm(gg, ff_locs)).view(batch_size, (nof_steps + 0*1), nof_targs, loc_vector_dim)
        all_ts_avg_prts_vels = torch.squeeze(torch.bmm(gg, ff_vels)).view(batch_size, (nof_steps + 0*1), nof_targs, loc_vector_dim)
        all_ts_avg_prts = np.zeros((batch_size, (nof_steps + 0*1), nof_targs, state_vector_dim))
        all_ts_avg_prts[:, :, :, (0, 2)] = all_ts_avg_prts_locs.cpu().detach().numpy()
        all_ts_avg_prts[:, :, :, (1, 3)] = all_ts_avg_prts_vels.cpu().detach().numpy()
        if x_with_t0 is not None:
            #ass2true_map = self.get_ass2true_map(x.cpu().detach().numpy(), all_ts_avg_prts)
            ass2true_map = self.get_ass2true_map(x_with_t0.cpu().detach().numpy(), all_ts_avg_prts)
        else:
            ass2true_map = None
        #print(ass2true_map)
        return all_ts_avg_prts_locs, all_ts_avg_prts_vels, ass2true_map

    def get_all_ts_torch_avg_particles_maapped(self, all_ts_avg_prts_locs, all_ts_avg_prts_vels, ass2true_map):
        batch_size, nof_steps, nof_targs, loc_vector_dim = all_ts_avg_prts_locs.shape
        batch_indcs = torch.tile(torch.reshape(torch.from_numpy(np.arange(batch_size)).to(self.device), (batch_size, 1, 1)), (1, nof_steps, nof_targs)).to(torch.long)
        ts_indcs = torch.tile(torch.reshape(torch.from_numpy(np.arange(nof_steps)).to(self.device), (1, nof_steps, 1)), (batch_size, 1, nof_targs)).to(torch.long)
        all_ts_avg_prts_locs_mapped = all_ts_avg_prts_locs[batch_indcs, ts_indcs, torch.from_numpy(ass2true_map).to(self.device).to(torch.long)]
        all_ts_avg_prts_vels_mapped = all_ts_avg_prts_vels[batch_indcs, ts_indcs, torch.from_numpy(ass2true_map).to(self.device).to(torch.long)]
        return all_ts_avg_prts_locs_mapped, all_ts_avg_prts_vels_mapped

    def get_all_ts_avg_particles_mapped(self, x_with_t0):
        all_ts_avg_prts_locs, all_ts_avg_prts_vels, ass2true_map = self.get_all_ts_avg_particles_and_mapping(x_with_t0)
        all_ts_avg_prts_locs_mapped, all_ts_avg_prts_vels_mapped = self.get_all_ts_torch_avg_particles_maapped(all_ts_avg_prts_locs, all_ts_avg_prts_vels, ass2true_map)
        return all_ts_avg_prts_locs_mapped, all_ts_avg_prts_vels_mapped

    def sav_intermediates(self,ts_idx, intermediates):
        self.nn1_lnw_in_per_iter[:, ts_idx]             = intermediates[0]
        self.nn1_lnw_out_per_iter[:, ts_idx]            = intermediates[1]
        self.nn3_in_full_parts_weights_var[:, ts_idx]   = intermediates[2]
        self.nn3_in_full_parts_weights[:, ts_idx]       = intermediates[3]
        self.nn3_in_full_parts_lnw[:, ts_idx]           = intermediates[4]
        self.nn3_in_unscaled_parts_locs[:, ts_idx]      = intermediates[5]

    def sav_batch_data(self, ts_idx, ln_weights, prts_locs, prts_vels, parents_incs):
        self.weights_per_iter[:, ts_idx] = torch.softmax(ln_weights, dim=1)
        self.lnw_per_iter[:, ts_idx] = ln_weights
        self.prts_locs_per_iter[:,ts_idx] = prts_locs
        self.prts_vels_per_iter[:,ts_idx] = prts_vels
        self.parents_incs_per_iter[:,ts_idx] = parents_incs

    def get_batch_data(self, ts_idx):
        ln_weights = self.lnw_per_iter[:, ts_idx]
        prts_locs = self.prts_locs_per_iter[:,ts_idx]
        prts_vels = self.prts_vels_per_iter[:,ts_idx]
        parents_incs = self.parents_incs_per_iter[:,ts_idx]
        return ln_weights, prts_locs, prts_vels, parents_incs

    def paint_nn3_wts_before_and_after(self, nof_ts_to_print, ts_jumps, max_wt):
        fontsize0 = 7
        fig, axs = plt.subplots(4, nof_ts_to_print, figsize=(15, 6))
        axs = axs.reshape((4,nof_ts_to_print))
        relevant_weigts = self.weights_per_iter[0, ts_jumps * np.arange(nof_ts_to_print)]
        # max_wt = np.max(relevant_weigts.cpu().detach().numpy())

        for ts_idx in np.arange(nof_ts_to_print):
            # plt.sca(axes[1, 1])
            curr_ts = 1 + ts_jumps * ts_idx
            hist_before, bin_edges_before = np.histogram(self.nn3_in_full_parts_weights[0, curr_ts].cpu().detach().numpy(), bins=20, range=(0, max_wt))
            axs[0, ts_idx].plot(np.sort(self.nn3_in_full_parts_weights[0, curr_ts].cpu().detach().numpy()))
            axs[0, ts_idx].set_title("ts " + str(curr_ts) + " wts be4 nn3", fontsize=fontsize0)
            axs[0, ts_idx].set_ylim(0, 0.05)
            axs[1, ts_idx].plot(bin_edges_before[:-1], hist_before)
            axs[1, ts_idx].set_title("ts " + str(curr_ts) + " wgts be4 nn3 hist", fontsize=fontsize0)
            axs[1, ts_idx].set_ylim(0, 100)

            hist, bin_edges = np.histogram(self.weights_per_iter[0, curr_ts].cpu().detach().numpy(), bins=20, range=(0, max_wt))
            axs[2, ts_idx].plot(np.sort(self.weights_per_iter[0, curr_ts].cpu().detach().numpy()))
            axs[2, ts_idx].set_title("ts " + str(curr_ts) + " wts aftr nn3", fontsize=fontsize0)
            axs[2, ts_idx].set_ylim(0, 0.05)
            axs[3, ts_idx].plot(bin_edges[:-1], hist)
            axs[3, ts_idx].set_title("ts " + str(curr_ts) + " wts aftr hist", fontsize=fontsize0)
            axs[3, ts_idx].set_ylim(0, 100)

            # axs[1, 0].imshow(ip[0].cpu().detach().numpy())
            # axs[1, 1].imshow(sip[0].cpu().detach().numpy())
        # plt.subplots_adjust(top=0.99, bottom=0.01, hspace=1.5, wspace=0.4)
        # fig.tight_layout(h_pad=50, w_pad=50)
        plt.show(block=False)

