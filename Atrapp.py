
import random
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal  as torch_mulvar_norm
from torch.distributions.normal import Normal  as torch_normal
#from models import *
import torchvision.transforms as torch_transforms
import time as time

def update_atrapp_torch5(opt, args, mm, sm, device, records, force_dont_sample_s1=False):
    do_checks = 1

    old_prts_locs, old_prts_vels, ln_old_weights, z_for_meas, is_first_sample_mu, is_TRAPP , F, Q, ts_idx = args

    batch_size, nof_parts, nof_targs, state_loc_vector_dim   = old_prts_locs.shape
    state_vector_dim = 4
    dt = opt.sensor_params.dt
    sensor_size = opt.sensor_params.sensor_size
    nof_s_x = opt.sensor_params.nof_s_x
    nof_s_y = opt.sensor_params.nof_s_y
    center = opt.sensor_params.center
    snr0db = opt.sensor_params.snr0db
    d0 = opt.sensor_params.d0
    eps = opt.sensor_params.eps
    assert len(old_prts_locs.shape) == 4
    #print(torch.abs(torch.sum(torch.exp(ln_old_weights))))
    #if not torch.abs(torch.sum(torch.exp(ln_old_weights)) - ln_old_weights.shape[0]) <=  1e-8:
    #    fff = 7
    #assert torch.abs(torch.sum(torch.exp(ln_old_weights)) - ln_old_weights.shape[0]) <=  1e-8
    assert z_for_meas is not None
    ################# functions ####################
    def tile_indces_batch(indcs, batch_size=batch_size, nof_parts=nof_parts):
        return torch.tile(torch.reshape(torch.from_numpy(indcs).to(device), (batch_size, -1)), (1, nof_parts)).to(torch.long)
    def tile_indces_batch2(indcs, batch_size=batch_size, nof_parts=nof_parts, nof_targs=nof_targs):
        return torch.tile(torch.reshape(torch.from_numpy(indcs).to(device), (batch_size, 1,1)), (1, nof_parts,nof_targs)).to(torch.long)
    def tile_float_batch_torch(indcs, batch_size=batch_size, nof_parts=nof_parts):
        return torch.tile(torch.reshape(indcs, (batch_size, -1)), (1, nof_parts))
    def tile_float_parts_torch2(parts, batch_size=batch_size, nof_parts=nof_parts, nof_targs=nof_targs):
        return torch.tile(torch.reshape(parts, (batch_size, 1, nof_targs)), (1, nof_parts, 1))
    def tile_indces_particles(batch_size=batch_size, nof_parts=nof_parts):
        return torch.tile(torch.reshape(torch.from_numpy(np.arange(nof_parts)).to(device), (1, nof_parts)), (batch_size, 1)).to(torch.long)
    def tile_indces_targ2(batch_size=batch_size, nof_parts=nof_parts, nof_targs=nof_targs):
        return torch.tile(torch.reshape(torch.from_numpy(np.arange(nof_targs)).to(device), (1, 1, nof_targs)), (batch_size, nof_parts, 1)).to(torch.long)



    def get_X_hat_tiled(prts_locs, prts_vels, ln_weights, is_first_sample_mu=is_first_sample_mu,F=F,Q=Q, batch_size=batch_size, nof_parts=nof_parts, nof_targs=nof_targs, state_loc_vector_dim=state_loc_vector_dim):
        new_prts_locs_for_X_hat, new_prts_vels_for_X_hat = mm.advance_locations(opt, is_first_sample_mu, prts_locs, prts_vels, device=device)
        assert new_prts_vels_for_X_hat.requires_grad == False
        weights = torch.softmax(ln_weights, dim=1)
        #weights1 = torch.exp(ln_weights)
        #weights1 = weights1/tile_float_batch_torch(torch.sum(weights1,dim=1))
        #assert torch.sum(torch.abs(weights - weights1)) <= 1e-8

        weighted_avg_loc = torch.bmm(weights.view(batch_size, 1, nof_parts), new_prts_locs_for_X_hat.reshape(batch_size, nof_parts, -1)).view(batch_size,1, nof_targs, state_loc_vector_dim)
        X_hat_loc = weighted_avg_loc #/ torch.tile(torch.reshape(torch.sum(weights, dim=1), (batch_size, 1, 1)), (1, nof_targs, state_loc_vector_dim))
        #X_hat = X_hat.detach()
        X_locs_hat_tiled = torch.tile(X_hat_loc, (1,nof_parts, 1, 1))# detached(x_hat) = true

        weighted_avg_vel = torch.bmm(weights.view(batch_size, 1, nof_parts), new_prts_vels_for_X_hat.reshape(batch_size, nof_parts, -1)).view(batch_size,1, nof_targs, state_loc_vector_dim)
        X_vels_hat_tiled = torch.tile(weighted_avg_vel, (1,nof_parts, 1, 1))

        return X_locs_hat_tiled, X_vels_hat_tiled

    def get_Xmj_hat(targ_idxs, Xkp1_loc_hat_tiled, Xkp1_vel_hat_tiled, batch_size=batch_size, nof_parts=nof_parts, nof_targs=nof_targs, state_loc_vector_dim=state_loc_vector_dim):
        Xmj_loc_hat = torch.zeros((batch_size, nof_parts, nof_targs-1, state_loc_vector_dim), device=device)
        Xmj_vel_hat = torch.zeros((batch_size, nof_parts, nof_targs-1, state_loc_vector_dim), device=device)
        for traj_idx in np.arange(batch_size):
            targ_idx = targ_idxs[traj_idx]
            targs_indcs = (*np.arange(targ_idx), *np.arange(targ_idx + 1, nof_targs))
            Xmj_loc_hat[traj_idx] = Xkp1_loc_hat_tiled[traj_idx, :, ((targs_indcs))]
            Xmj_vel_hat[traj_idx] = Xkp1_vel_hat_tiled[traj_idx, :, ((targs_indcs))]
        return Xmj_loc_hat, Xmj_vel_hat

    def get_torch_mn_samples(ln_targs_weights, batch_size=batch_size, nof_parts=nof_parts, nof_targs=nof_targs):
        targs_weights = torch.softmax(ln_targs_weights, dim=1).detach()
        #targs_weights1 = torch.exp(ln_targs_weights.detach())
        #targs_weights1 /= torch.tile(torch.reshape(torch.sum(targs_weights1, dim=1), (batch_size, 1, nof_targs)), (1, nof_parts, 1))
        #assert torch.sum(torch.isnan(targs_weights))==0
        #assert torch.sum(torch.abs(targs_weights-targs_weights1)) <= 1e-8
        sampled_indcs_per_targ = torch.zeros((batch_size, nof_parts, nof_targs), device=device).to(torch.long)
        for traj_idx in np.arange(batch_size):
            for targ_idx in np.arange(nof_targs):
                sampled_indcs_per_targ[traj_idx, :, targ_idx] = torch.multinomial(targs_weights[traj_idx, :, targ_idx], nof_parts, replacement=True).to(torch.long)
        return sampled_indcs_per_targ

    def log_lh_normalize(log_lh):
        return log_lh - tile_float_batch_torch(torch.max(log_lh, dim=1).values).detach() - 1
    def log_lh_normalize2(weights):
        return weights - tile_float_parts_torch2(torch.max(weights, dim=1).values).detach()-1
    def original_order(ordered, indices):
        return ordered.gather(1, indices.argsort(1))

    def grad_on_parts_get_parts_adj(curr_parts_locs, curr_wts):
        # (1) dl/dwj = (dl/dxj * dxj/dwj) + (dl/dyj * dyj/dwj)
        # sj = softmax(wj)
        # dl/dxj -> dl/dxj * (x1*ds1/dwj +..+ xN*dsN/dwj) /(sj*xj)
        # dl/dxj -> dl/dxj * (1- (x1*s1 +..+ xN*sN)/xj)
        debug_wts_update = torch.zeros_like(curr_parts_locs, requires_grad=False).detach()
        # sm_ln_wts = torch.softmax(new_ln_weights_post_nn1, dim=1).detach()
        sm_ln_wts = torch.softmax(curr_wts, dim=1).detach()
        # sm_ln_wts = torch.softmax(new_ln_weights_post_nn1, dim=1)[tiled_indces_batch2, sampled_indcs_app_targ.to(torch.long), tiled_indces_targ2].detach()
        ff = torch.permute(curr_parts_locs, (0, 2, 3, 1)).reshape(batch_size * nof_targs, state_loc_vector_dim, nof_parts)
        # xj = xj*exp(wj-wj.detach)
        # dxj/dwj = xj*exp(wj-wj.detach) = xj
        # xj=0 => dxj/dwj=0 => (1) dl/dxj * dxj/dwj = 0 => so dl/dxj (from curr_parts.gradient) doesnt matter so changing it to not divide by 0
        #ff_no_zeros = torch.where(ff != 0, ff, 1e-1000)
        gg = torch.permute(sm_ln_wts, (0, 2, 1)).reshape(batch_size * nof_targs, nof_parts, 1)
        # debug_wts_update = torch.permute((1 - torch.bmm(ff, gg) / ff_no_zeros).reshape(batch_size, nof_targs, 2, nof_parts), (0, 3, 1, 2))
        debug_wts_update = torch.permute((1 - torch.bmm(ff, gg) / ff).reshape(batch_size, nof_targs, 2, nof_parts), (0, 3, 1, 2))
        #debug_wts_update = torch.permute((1 - torch.bmm(ff, gg) / ff_no_zeros).reshape(batch_size, nof_targs, 2, nof_parts), (0, 3, 1, 2))
        return debug_wts_update

    def grad_on_parts_get_parts_adj_samp(curr_parts_locs, curr_wts, parents_indces):
        device0 = 'cuda'
        sampled_indcs_app_targ_cpu = parents_indces.to(device0)
        bins_count = torch.zeros((batch_size * nof_targs, nof_parts)).to(device0)
        flattened = torch.reshape(torch.transpose(sampled_indcs_app_targ_cpu, 1, 2), (batch_size * nof_targs, nof_parts))
        for idx, parts_set in enumerate(flattened):
            bins_count[idx] = torch.histc(parts_set.to(torch.float), bins=nof_parts, min=0, max=nof_parts)  # all_targs_picked_old_prts_locs_graded.retain_grad()
        bins_count = torch.transpose(torch.reshape(bins_count, (batch_size, nof_targs, nof_parts)), 1, 2).to(device0)
        nonz_indcs = torch.nonzero(bins_count)
        xy_sum_of_belongs = torch.zeros_like(curr_parts_locs, device=device0, requires_grad=False)
        xy_sum_of_not_bel = torch.zeros_like(curr_parts_locs, device=device0, requires_grad=False)
        for idx, (set_idx, part_idx, targ_idx) in enumerate(nonz_indcs):
            indcs = torch.stack(list(torch.where(sampled_indcs_app_targ_cpu[set_idx, :, targ_idx] == part_idx)), dim=0).to(torch.long)
            len_indcs = indcs.shape[-1]
            xy_sum_of_belongs[set_idx, indcs, targ_idx] += torch.sum(torch.reshape(curr_parts_locs[set_idx, indcs, targ_idx], (1, len_indcs, 1,2)),dim=1)
            xy_sum_of_not_bel[set_idx, indcs, targ_idx] += torch.sum(torch.reshape(curr_parts_locs[set_idx, :, targ_idx],(1, nof_parts, 1,2)), dim=1) - xy_sum_of_belongs[set_idx, indcs, targ_idx]#torch.sum(torch.reshape(curr_parts_locs[set_idx, indcs, targ_idx],(1, len_indcs, 1,2)), dim=1)
            #wts_weighting[set_idx, indcs, targ_idx] = 1 / bins_count[set_idx, part_idx, targ_idx]
        debug_wts_update = (nof_parts - 1)/nof_parts - 1/nof_parts*xy_sum_of_not_bel/xy_sum_of_belongs
        # softmax_ln_wts = torch.softmax(new_ln_weights_post_nn1, dim=1).detach()
        return debug_wts_update

    ################################################################################
    start_time = time.time()
    measure_time_counter = 0
    nn3_time_counter = 0
    normalize_bj_mu_bet_parts = False # gives eaqual weight to each target (not done on alg)
    assert len(z_for_meas.shape) == 3
    tiled_batch_indcs = tile_indces_batch(np.arange(batch_size))  # torch.tile(torch.reshape(torch.arange(batch_size).to(device), (batch_size, -1)), (1, nof_parts)).to(torch.long)
    tiled_part_indcs = tile_indces_particles().detach()
    tiled_indces_batch2 = tile_indces_batch2(np.arange(batch_size))
    tiled_indces_targ2 = tile_indces_targ2().detach()
    # ============= making Xkp1_loc_hat_tiled, Xkp1_vel_hat_tiled ============= #
    #ln_old_weights = log_lh_normalize(ln_old_weights)
    Xkp1_loc_hat_tiled, Xkp1_vel_hat_tiled = get_X_hat_tiled(old_prts_locs, old_prts_vels, ln_old_weights)
    #active_sensors_mask = sm.get_active_sensors_mask_from_old_parts_old(opt, mm, old_prts_locs, old_prts_vels, ln_old_weights, F, Q, device) # there was a bug
    active_sensors_mask = sm.get_active_sensors_mask_from_old_parts(opt, mm, old_prts_locs, old_prts_vels, ln_old_weights, device) # there was a bug

    # ---------- making X_hat_tiled end ------------ #
    new_ln_weights_targs = torch.zeros((batch_size, nof_parts, nof_targs), device=device)
    ln_b_mu          = torch.zeros((batch_size, nof_parts, nof_targs), device=device)
    targets_order     = np.zeros((batch_size, nof_targs))
    for traj_idx in np.arange(batch_size):
        targets_order[traj_idx] = np.random.permutation(np.arange(nof_targs)).astype(np.int)
    # ============= A_1 start ============= #
    new_prts_locs0, new_prts_vels0  = mm.advance_locations(opt, is_first_sample_mu, old_prts_locs, old_prts_vels, device=device, print_seed=False)  # detached(old_prts_locs)=true
    #new_prts_locs0[:, :, :, (1, 3)] = new_prts_locs0[:, :, :, (1, 3)].detach()
    # ------------- A_1 end ----------------#
    # ============= M_1 start ============= #
    for targ_idxs in np.transpose(targets_order):
        tiled_curr_targs_indcs = tile_indces_batch(targ_idxs)
        curr_target_new_prts_locs0 = torch.unsqueeze(new_prts_locs0[tiled_batch_indcs, tiled_part_indcs, tiled_curr_targs_indcs], -2) # etached(new_prts_locs0)=true
        curr_target_new_prts_vels0 = torch.unsqueeze(new_prts_vels0[tiled_batch_indcs, tiled_part_indcs, tiled_curr_targs_indcs], -2) # etached(new_prts_locs0)=true
        Xmj_loc_hat, Xmj_vel_hat = get_Xmj_hat(targ_idxs, Xkp1_loc_hat_tiled, Xkp1_vel_hat_tiled)
        curr_X_loc_hat = torch.cat((Xmj_loc_hat, curr_target_new_prts_locs0), dim=2) # etached(Xmj_hat, curr_target_new_prts_locs0)=true
        #curr_X_vel_hat = torch.cat((Xmj_vel_hat, curr_target_new_prts_vels0), dim=2)
        measure_start_time = time.time()
        ln_bj_mu = sm.get_lh_measure_prts_locs_with_measurement_torch(opt, curr_X_loc_hat, z_for_meas, active_sensors_mask, return_log=True, device=device)# detached(curr_X_hat, z_for_meas)=true
        measure_time_counter += time.time() - measure_start_time
        ln_bj_mu = log_lh_normalize(ln_bj_mu)
        if normalize_bj_mu_bet_parts:
            ln_bj_mu = ln_bj_mu - torch.log(torch.tile(torch.reshape(torch.sum(torch.exp(ln_bj_mu), dim=1), (batch_size, 1)), (1, nof_parts)))
        ln_b_mu[tiled_batch_indcs, tiled_part_indcs, tiled_curr_targs_indcs] = ln_bj_mu
        #new_ln_weights_targs[tiled_batch_indcs, tiled_part_indcs, tiled_curr_targs_indcs] = torch.multiply(old_weights, bj_mu) # etached = false

    #new_ln_weights_targs = torch.unsqueeze(ln_old_weights, -1)+ ln_b_mu  # etached = false
    new_ln_weights_targs =  ln_b_mu  # etached = false
    # ------------- M_1 end ----------------#
    #============= NN_1 start =============#
    # for the entrence of nn1, normalize such that for each batch the biggest weight is ln(0)
    new_ln_weights_targs_nn1_in = log_lh_normalize2(new_ln_weights_targs)

    new_ln_weights_pre_s1 = new_ln_weights_targs_nn1_in + torch.unsqueeze(ln_old_weights, -1)
    new_ln_weights_post_nn1_out = torch.clone(new_ln_weights_pre_s1)
    #============= S_1 start =============#
    #for sampling, changing to weights and normalizing to have sum=1

    sampled_indcs_app_targ = get_torch_mn_samples(new_ln_weights_pre_s1.detach()).to(torch.long)
    #################################################
    picked_2b_advanced_locs = torch.reshape(old_prts_locs[tiled_indces_batch2, sampled_indcs_app_targ, tiled_indces_targ2], (batch_size, nof_parts, nof_targs, state_loc_vector_dim))  # detached=false
    picked_2b_advanced_vels = torch.reshape(old_prts_vels[tiled_indces_batch2, sampled_indcs_app_targ, tiled_indces_targ2], (batch_size, nof_parts, nof_targs, state_loc_vector_dim))
    x_star_loc, x_star_vel = mm.advance_locations(opt, False, picked_2b_advanced_locs, picked_2b_advanced_vels, device=device)  # detached=false
    new_ln_weights_post_nn1_picked = new_ln_weights_pre_s1[tiled_indces_batch2, sampled_indcs_app_targ, tiled_indces_targ2]

    ##################################################
    #------------- A_2 end ----------------#\
    if not is_TRAPP:
        all_target_new_prts_locs = x_star_loc # detached=false\
        all_target_new_prts_vels = x_star_vel # detached=false\
        parents_indces = sampled_indcs_app_targ
        ln_bj_x_kp1 = ln_b_mu[tiled_indces_batch2, sampled_indcs_app_targ.to(torch.long), tiled_indces_targ2]  # detached(bj_mu)=true
    else:
        # ============= M_2 start ============= #
        ln_b_x_star = torch.zeros((batch_size, nof_parts,nof_targs), device=device)
        for targ_idxs in np.transpose(targets_order):
            Xmj_loc_hat, Xmj_vel_hat  = get_Xmj_hat(targ_idxs, Xkp1_loc_hat_tiled) # TODO make sure indeed same Xkp1_loc_hat_tiled
            tiled_curr_targs_indcs = tile_indces_batch(targ_idxs)
            curr_X_loc_hat = torch.cat((Xmj_loc_hat, torch.reshape(x_star_loc[tiled_batch_indcs, tiled_part_indcs, tiled_curr_targs_indcs], (batch_size, nof_parts, 1, state_loc_vector_dim))), dim=2)
            measure_start_time = time.time()
            bj_x_star_log = sm.get_lh_measure_prts_locs_with_measurement_torch(opt, curr_X_loc_hat, z_for_meas, active_sensors_mask, return_log=True, device=device)
            measure_time_counter += time.time() - measure_start_time
            bj_x_star_log = log_lh_normalize(bj_x_star_log)
            ln_b_x_star[tiled_batch_indcs, tiled_part_indcs, tiled_curr_targs_indcs] = bj_x_star_log

        ln_r_x_star = ln_b_x_star - ln_b_mu[tiled_indces_batch2, sampled_indcs_app_targ.to(torch.long), tiled_indces_targ2]
        # ------------- M_2 end ----------------#
        # ============= NN_2 start ============= #
        # for the entrence of nn1, normalize such that for each batch the biggest weight is ln(0) (used to be sum of weights=1
        ln_r_x_star = log_lh_normalize2(ln_r_x_star)
        # ------------- NN_2 end ----------------#
        # ============= S_2 start ============= #
        sampled_indcs_trapp_targ = get_torch_mn_samples(ln_r_x_star)
        all_target_new_prts_locs = x_star_loc[tiled_indces_batch2, sampled_indcs_trapp_targ, tiled_indces_targ2]
        all_target_new_prts_vels = x_star_vel[tiled_indces_batch2, sampled_indcs_trapp_targ, tiled_indces_targ2]
        ln_bj_x_kp1 = ln_b_x_star[tiled_indces_batch2, sampled_indcs_trapp_targ, tiled_indces_targ2]
        parents_indces = sampled_indcs_app_targ[tiled_indces_batch2, sampled_indcs_trapp_targ, tiled_indces_targ2]
        # ------------- S_2 end ----------------#
    new_prts_locs = all_target_new_prts_locs# detached=false
    new_prts_vels = all_target_new_prts_vels# detached=false

    if 1: # opt.skip_m2_to_nn3_flag=False
        #equal_graded_weights_s1 = ln_weights_s1 - ln_weights_s1.detach()
    # ============= M_3 start ============= #
        ln_pi_target_bj = torch.sum(ln_bj_x_kp1, dim=2)  # detached(ln_bj_x_kp1)=true
        ln_pi_target_bj = log_lh_normalize(ln_pi_target_bj)
        measure_start_time = time.time()
        m3_ln_new_parts_lh = sm.get_lh_measure_prts_locs_with_measurement_torch(opt, new_prts_locs, z_for_meas, active_sensors_mask, return_log=True, device=device)# detached(new_prts_locs)=false
        measure_time_counter += time.time() - measure_start_time
        # ------------- M_3 end ----------------#
        post_m3_ln_weights = m3_ln_new_parts_lh - ln_pi_target_bj.detach()  # +ln_weights_s1 - ln_weights_s1.detach()#detached(m3_ln_new_parts_lh=false, ln_pi_target_bj=true)=false

        # ============= NN_3 start =============#

        nn3_in_unscaled_parts_locs = new_prts_locs.detach()
        if 1:
            new_ln_weights_targs_nn3_in = log_lh_normalize(post_m3_ln_weights)
            nn3_in_full_parts_weights_var = torch.var(torch.softmax(new_ln_weights_targs_nn3_in.detach(), dim=1), unbiased=False, dim=1)
            nn3_in_full_parts_lnw = new_ln_weights_targs_nn3_in.detach()
            nn3_in_full_parts_weights = torch.softmax(new_ln_weights_targs_nn3_in.detach(), dim=1)

            new_prts_locs_post_nn3 = nn3_in_unscaled_parts_locs
            new_weights_post_nn3 = new_ln_weights_targs_nn3_in


        #TODO change to that
        ln_final_weights = log_lh_normalize(new_weights_post_nn3)


    #if opt.s1_grad_in_weights_flag:
    #    new_prts_locs_post_nn3 =  new_prts_locs_post_nn3.detach()
    #else:
    #    ln_final_weights =  ln_final_weights.detach()
    #ln_final_weights = -torch.ones_like(ln_final_weights)
    iteration_time =  time.time() - start_time
    #print("iteration time: " +str(iteration_time)+", measuretiem of it: " +str(measure_time_counter)+", which is: %.3f" % (measure_time_counter/iteration_time))
    return new_prts_locs_post_nn3, new_prts_vels, ln_final_weights, parents_indces, (new_ln_weights_targs_nn1_in, new_ln_weights_post_nn1_out, nn3_in_full_parts_weights_var, nn3_in_full_parts_weights, nn3_in_full_parts_lnw, nn3_in_unscaled_parts_locs), (iteration_time, nn3_time_counter, measure_time_counter)

update_atrapp_torch = update_atrapp_torch5#_test2