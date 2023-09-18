import numpy as np
import torch

class BatchData(object):
    def __init__(self, batch_size, nof_steps, nof_parts, nof_targs, device):
        self.device = device
        self.nof_parts = nof_parts
        self.reset(batch_size, nof_steps, nof_parts, nof_targs, device)

    def reset(self, batch_size, nof_steps, nof_parts, nof_targs, device):
        self.prts_locs_per_iter = torch.zeros((batch_size, nof_steps, self.nof_parts,nof_targs, 2), device=device)
        self.prts_vels_per_iter = torch.zeros((batch_size, nof_steps, self.nof_parts,nof_targs, 2), device=device)
        self.weights_per_iter = torch.zeros((batch_size, nof_steps , self.nof_parts), device=device)
        self.lnw_per_iter = torch.zeros((batch_size, nof_steps , self.nof_parts), device=device)
        self.parents_incs_per_iter = torch.zeros((batch_size, nof_steps , self.nof_parts,nof_targs), device=device)
        # not in use
        self.nn1_lnw_in_per_iter = torch.zeros((batch_size, nof_steps , self.nof_parts,nof_targs), device=device)
        self.nn1_lnw_out_per_iter = torch.zeros((batch_size, nof_steps , self.nof_parts,nof_targs), device=device)
        self.nn3_in_full_parts_weights_var = torch.zeros((batch_size, nof_steps ), device=device)
        self.nn3_in_full_parts_weights = torch.zeros((batch_size, nof_steps , self.nof_parts), device=device)
        self.nn3_in_full_parts_lnw = torch.zeros((batch_size, nof_steps , self.nof_parts), device=device)
        self.nn3_in_unscaled_parts_locs = torch.zeros((batch_size, nof_steps , self.nof_parts,nof_targs, 2), device=device)

    def get_trajs_tensors(self, trajs_to_advance):
        batch_size, nof_steps, nof_parts, nof_targs, _ = self.prts_locs_per_iter.shape
        new_bd = BatchData(trajs_to_advance.shape[0], nof_steps, nof_parts, nof_targs, self.device)
        new_bd.prts_locs_per_iter = self.prts_locs_per_iter[trajs_to_advance]
        new_bd.prts_vels_per_iter = self.prts_vels_per_iter[trajs_to_advance]
        new_bd.weights_per_iter = self.weights_per_iter[trajs_to_advance]
        new_bd.lnw_per_iter = self.lnw_per_iter[trajs_to_advance]
        new_bd.parents_incs_per_iter = self.parents_incs_per_iter[trajs_to_advance]
        new_bd.nn1_lnw_in_per_iter = self.nn1_lnw_in_per_iter[trajs_to_advance]
        new_bd.nn1_lnw_out_per_iter = self.nn1_lnw_out_per_iter[trajs_to_advance]
        new_bd.nn3_in_full_parts_weights_var = self.nn3_in_full_parts_weights_var[trajs_to_advance]
        new_bd.nn3_in_full_parts_weights = self.nn3_in_full_parts_weights[trajs_to_advance]
        new_bd.nn3_in_full_parts_lnw = self.nn3_in_full_parts_lnw[trajs_to_advance]
        new_bd.nn3_in_unscaled_parts_locs = self.nn3_in_unscaled_parts_locs[trajs_to_advance]
        return new_bd

    def set_trajs_tensors(self, old_bd, trajs_to_advance):
        batch_size, nof_steps, nof_parts, nof_targs, _ = self.prts_locs_per_iter.shape
        self.prts_locs_per_iter[trajs_to_advance] = old_bd.prts_locs_per_iter
        self.prts_vels_per_iter[trajs_to_advance] = old_bd.prts_vels_per_iter
        self.weights_per_iter[trajs_to_advance] = old_bd.weights_per_iter
        self.lnw_per_iter[trajs_to_advance] = old_bd.lnw_per_iter
        self.parents_incs_per_iter[trajs_to_advance] = old_bd.parents_incs_per_iter
        self.nn1_lnw_in_per_iter[trajs_to_advance] = old_bd.nn1_lnw_in_per_iter
        self.nn1_lnw_out_per_iter[trajs_to_advance] = old_bd.nn1_lnw_out_per_iter
        self.nn3_in_full_parts_weights_var[trajs_to_advance] = old_bd.nn3_in_full_parts_weights_var
        self.nn3_in_full_parts_weights[trajs_to_advance] = old_bd.nn3_in_full_parts_weights
        self.nn3_in_full_parts_lnw[trajs_to_advance] = old_bd.nn3_in_full_parts_lnw
        self.nn3_in_unscaled_parts_locs[trajs_to_advance] = old_bd.nn3_in_unscaled_parts_locs

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

    def get_all_ts_avg_particles_and_mapping(self):
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
        return all_ts_avg_prts_locs, all_ts_avg_prts_vels

    def sav_batch_data(self, ts_idx, ln_weights, prts_locs, prts_vels, parents_incs):
        self.weights_per_iter[:, ts_idx] = torch.softmax(ln_weights, dim=1)
        self.lnw_per_iter[:, ts_idx] = ln_weights
        self.prts_locs_per_iter[:,ts_idx] = prts_locs
        self.prts_vels_per_iter[:,ts_idx] = prts_vels
        self.parents_incs_per_iter[:,ts_idx] = parents_incs

    def get_batch_data(self, ts_idx):
        if ts_idx is not None:
            ln_weights = self.lnw_per_iter[:, ts_idx]
            prts_locs = self.prts_locs_per_iter[:,ts_idx]
            prts_vels = self.prts_vels_per_iter[:,ts_idx]
            parents_incs = self.parents_incs_per_iter[:,ts_idx]
        else:
            ln_weights = self.lnw_per_iter
            prts_locs = self.prts_locs_per_iter
            prts_vels = self.prts_vels_per_iter
            parents_incs = self.parents_incs_per_iter
        return ln_weights, prts_locs, prts_vels, parents_incs
