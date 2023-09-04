""" Particle filters

@author: Jerker Nordh
"""

import numpy
import math
import copy
import random
import numpy as np
import torch
import datetime as datetime


import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from AtrappModel import AtrappModel
from BatchData import *
import torch.nn.functional as F

import matplotlib.pyplot as plt

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


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

        return np.min(zs)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)


class PfBatchHandler(object):
    def __init__(self, model, opt):#batch_size, nof_particles, nof_timesteps, nof_targs):

        self.model = model
        self.device = self.model.device
        self.opt = opt
        self.train_nn1 = True
        self.train_nn2 = True
        self.train_nn3 = True
        self.loss_type = 'none'
    def get_desired_x_detached(self, x_with_t0, all_ts_avg_prts_locs, ratio, device):
        assert ratio <= 1 and ratio >= 0, "fsewsetsets"
        x_tag = torch.zeros_like(x_with_t0, device=device)
        all_ts_avg_prts_locs_tag = all_ts_avg_prts_locs  #
        x_tag[:, :, :, 0] = all_ts_avg_prts_locs_tag[:, :, :, 0]
        x_tag[:, :, :, 2] = all_ts_avg_prts_locs_tag[:, :, :, 1]
        x_to_take = ratio * x_with_t0 + (1 - ratio) * x_tag
        return x_to_take.detach()

    def clear_db(self, x_with_t0):
        batch_size, nof_steps, nof_targs, _ = x_with_t0.shape
        #self.bd = BatchData(batch_size, 0, self.opt.nof_parts, nof_targs, self.device)
        self.traj = []
        if 1:#self.opt.add_loss_type == 'heatmap' and self.opt.heatmap_use_ref:
            self.bd_list_for_paint = []

    #def create_initial_estimate(self, x0, v0):
    #    return torch.from_numpy(self.model.create_initial_estimate(x0, v0, self.opt.nof_parts))


    def get_ospa_dist(self, true_x, ess_x, p, c, use_ratio=False):
        batch_size, nof_steps, nof_targs, _ = true_x.shape
        dists = torch.sqrt(
            torch.pow(true_x[:, :, :, 0] - ess_x[:, :, :, 0], 2) +
            torch.pow(true_x[:, :, :, 2] - ess_x[:, :, :, 1], 2)
        )
        # temp = torch.minimum(torch.tensor(snr0db), temp0)
        minimum_dist_c = torch.where(dists <= c, dists, c)

        ospa_batch = torch.pow(
            1 / nof_targs * torch.sum(
                torch.pow(minimum_dist_c, p
                          ) + 0, dim=2
            ), 1 / p
        )
        return ospa_batch


    def get_batch_loss(self,x, z):
        def prts_locs_per_iter_hook(grad):
            print("prts_locs_per_iter_hook: "+str(grad))
        def weights_per_iter_hook(grad):
            print("weights_per_iter_hook: "+str(grad))
        def all_ts_avg_prts_locs_hook(grad):
            print("all_ts_avg_prts_locs_hook: "+str(grad))

        """
        Append new time step to trajectory

        Args:f
         - y (array-like): Measurement of x_{t+1}

        Returns:
         (bool) True if the particle approximation was resampled
        """

        batch_size, nof_steps, nof_targs, _ = x.shape
        loss_batch_ts = torch.zeros((batch_size, nof_steps+1), device=self.device)
        loss_batch_ts_list = []
        ospa_batch_ts = torch.zeros((batch_size, nof_steps+1), device=self.device)
        ospa_batch_ts.requires_grad = False
        # output || time |  time    |   real   | yi ([]=db idx) | pi (parents) |part(start->end)| w(start->end)  | traj  |
        # idx (i)|| idx  |start->end|   state  | final on idx=i |              |final on idx=i  | final on idx=i |  len  |
        #===================================================================================================================== |
        #    0   ||  0   |  -1->0   |  None    | y0=None        |  p0= None    | None->random   |   None->w0=1   |  0->1 |
        #    1   ||  1   |   0->1   |  x_db[0] | y1=y_db[0]     |  p1= part0   | part0->part1   |   w0->w1       |  1->2 |
        #    2   ||  2   |   1->2   |  x_db[1] | y2=y_db[1]     |  p2= part1   | part1->part2   |   w1->w2       |  2->3 |
        #    3   ||  3   |   2->3   |  x_db[2] | y2=y_db[2]     |  p3= part2   | part2->part3   |   w2->w3       |  3->4 |
        #torch.autograd.set_detect_anomaly(True)

        self.model.reset_before_batch([False, True, False], x)
        self.clear_db(x)
        batch_size, nof_steps, nof_targs, _ = x.shape
        true_vels, true_locs, x0, v0 = self.get_cheats(x)
        x_with_t0 = torch.cat((x[:, 0:1], x), dim=1)
        state_vector_dim = x.shape[-1]

        #self.traj.append(curr_traj_step)
        curr_bd = BatchData(batch_size, 1, self.opt.nof_parts, nof_targs, self.device)
        atrapp_time = 0
        nn3_time = 0
        meas_time = 0
        for i in (-1+numpy.arange(z.shape[1]+1)):
            ts_idx = i + 1
            if i==-1:
                curr_measmnts = None
            else:
                curr_measmnts = z[:, i].to(self.device)
            curr_bd, curr_traj_step, timings = self.forward_one_time_step(old_bd=curr_bd, measmnts=curr_measmnts, ts_idx=ts_idx, x0=x0,v0=v0, true_vels=true_vels, true_locs=true_locs)
            curr_atrapp_time, curr_nn3_time, curr_meas_time = timings
            atrapp_time+=curr_atrapp_time
            nn3_time+=curr_nn3_time
            meas_time+=curr_meas_time

            self.traj.append(curr_traj_step)
            #########################################
            # TODO check x_with_t0[:,ts_idx:ts_idx+1] on ts_idx=2


            all_ts_avg_prts_locs, all_ts_avg_prts_vels, ass2true_map = curr_bd.get_all_ts_avg_particles_and_mapping(x_with_t0[:,ts_idx:ts_idx+1])
            all_ts_avg_prts_locs_mapped, all_ts_avg_prts_vels_mapped = curr_bd.get_all_ts_torch_avg_particles_maapped(all_ts_avg_prts_locs, all_ts_avg_prts_vels, ass2true_map)
            desired_x_curr_ts = copy.deepcopy(x_with_t0[:, ts_idx:ts_idx + 1])
            assert self.opt.nof_targs == 1
            x_for_ospa_loss = x_with_t0[:,ts_idx:ts_idx+1]
            ospa_loss_b_ts = self.get_ospa_dist(x_for_ospa_loss, all_ts_avg_prts_locs_mapped, self.opt.ospa_p, self.opt.ospa_c)
            loss_b_curr_ts = ospa_loss_b_ts


            #print("real x "+str(x))
            ospa_batch_curr_ts = self.get_ospa_dist(copy.copy(x_with_t0[:,ts_idx:ts_idx+1].detach()), all_ts_avg_prts_locs_mapped.detach(), self.opt.ospa_p, 10.)

            #loss_batch_ts_list.append(loss_b_curr_ts*self.opt.ospa_loss_mult + regul_loss)
            loss_batch_ts[:,ts_idx:ts_idx+1]  = loss_b_curr_ts
            ospa_batch_ts[:,ts_idx:ts_idx+1] = ospa_batch_curr_ts


        #print("atrapp time: %.3f "%(atrapp_time) + ", meas time of it: %.3f "%(meas_time) + ", which is: %.3f" % (meas_time /atrapp_time))
        return loss_batch_ts[:,1:], ospa_batch_ts[:,1:], (atrapp_time, nn3_time, meas_time)
        #return torch.stack(loss_batch_ts_list, dim=1)[:, 1:], ospa_batch_ts[:, 1:]

    def forward_one_time_step(self, old_bd, measmnts, ts_idx, x0, v0, true_vels, true_locs):
        assert ((measmnts is None) and (ts_idx == 0)) or ((measmnts is not None) and (not ts_idx == 0)), "only supports these scenrios"
        bd_new = old_bd.make_clean_new_bd()
        if (measmnts is None):
            timings = 0, 0, 0
            #particles0 = self.(x0, v0)
            particles0 = torch.from_numpy(self.model.create_initial_estimate(x0, v0, self.model.opt.nof_parts))

            pa = ParticleApproximation(particles=particles0)
            batch_size, _nof_tss, nof_targs, state_vector_dim = pa.part.shape
            # only ancestors initial for atrapp to change to put inside pa that is outputted
            ancestors = numpy.tile(numpy.reshape(numpy.arange(pa.nof_parts),(1, pa.nof_parts, 1)), (batch_size, 1, nof_targs))
            first_traj_elem = TrajectoryStep(pa, ancestors=ancestors)
            prts_locs = torch.from_numpy(first_traj_elem.pa.part[:,:,:,(0,2)]).to(self.model.device)
            prts_vels = torch.from_numpy(first_traj_elem.pa.part[:,:,:,(1,3)]).to(self.model.device)
            ln_weights = torch.from_numpy(first_traj_elem.pa.w).to(self.model.device)
            parents_incs = torch.from_numpy(first_traj_elem.ancestors).to(self.model.device)
            curr_traj_step = first_traj_elem
            # old_bd is empty updating only final outputs for bf_ref
            t0_nn1_out_lnw = torch.tile(torch.unsqueeze(ln_weights, -1), (1,1,nof_targs))
            t0_nn3_out_wts_var = torch.var(torch.softmax(ln_weights.detach(), dim=1), unbiased=False, dim=1)
            intermediates = t0_nn1_out_lnw, t0_nn1_out_lnw, t0_nn3_out_wts_var , torch.softmax(ln_weights, dim=-1), ln_weights, prts_locs
            old_bd.sav_batch_data(0, ln_weights, prts_locs, prts_vels, parents_incs)
        else:# not first ts
            if (self.loss_type == 'heatmap' and self.opt.heatmap_use_ref) or self.loss_type == 'sinkhorn':
                curr_seeds_states = self.sh.get_current_seeds_states()

            #measmnts = torch.from_numpy(measmnts)
            #new_ln_weights, new_parts_locs, new_parts_vels, new_parents_incs
            ln_weights, prts_locs, prts_vels,  parents_incs = old_bd.get_batch_data(0)
            prts_locs, prts_vels, ln_weights, parents_incs, intermediates, timings = self.model.forward(
                prts_locs=prts_locs,
                prts_vels=prts_vels,
                ln_weights=ln_weights,
                parents_incs=parents_incs,
                z_for_meas=measmnts,
                ts_idx=ts_idx,
                true_vels=true_vels, true_locs=true_locs
            )
            particles0 = np.zeros((*prts_locs.shape[:-1], 4))
            particles0[:, :, :, (0, 2)] = prts_locs.cpu().detach().numpy()
            particles0[:, :, :, (1, 3)] = prts_vels.cpu().detach().numpy()
            pa = ParticleApproximation(particles=particles0, logw=ln_weights.cpu().detach().numpy())
            curr_traj_step = TrajectoryStep(pa, ancestors=parents_incs.cpu().detach().numpy())

        bd_new.sav_intermediates(0, intermediates)
        bd_new.sav_batch_data(0, ln_weights, prts_locs, prts_vels, parents_incs)
        return bd_new, curr_traj_step, timings

    def get_cheats(self, x_for_cheats):
        if x_for_cheats.shape[1]>1:
            true_vels = (x_for_cheats[:, 1:, :, (0, 2)] - x_for_cheats[:, :-1, :, (0, 2)]) / self.opt.tau
        else:
            true_vels = torch.zeros_like(x_for_cheats[:, 0:1, :, (0, 2)])
        # true_vels[1:] = (x_k[2:, :, (0, 2)] - x_k[:-2, :, (0, 2)]) / (2*ms.tau)
        # true_vels = x_k[:,:,(1,3)]
        # true_vels[(0,1),:] = true_vels[2,:]
        # true_vels = true_vels[:,(1,0)]
        # true_vels[1:] = true_vels[:-1]
        get_next_vels = False
        if get_next_vels:  # use the velocity from k to k+1 , (that is in fact impossible)
            true_vels[:, :, :-1] = true_vels[:, :, 1:]
        true_locs = x_for_cheats[:, :, :, (0, 2)]
        true_x0 = true_locs[:, 0]
        if 0:
            true_v0 = true_vels[:, 0]
        else:
            true_v0 = x_for_cheats[:, 0, :, (1, 3)]
        return true_vels, true_locs, true_x0, true_v0

    def plot_3d_particle_traj(self, x_t_locs, x_t_vels, time_steps, ax=None, draw_line=True, draw_parts=True, draw_arrows=True):
        assert len(x_t_locs.shape) == 3
        center = self.opt.sensor_params.center
        sensor_size = self.opt.sensor_params.sensor_size
        do_show = False
        if ax == None:
            do_show = True
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            #ax = fig.add_subplot(projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('time')
            ax.set_xlim(self.opt.sensor_params.center[0] - sensor_size[0] / 2, center[0] + sensor_size[0] / 2)
            ax.set_ylim(center[1] - sensor_size[1] / 2, center[1] + sensor_size[1] / 2)
        (dc0, nof_targets, dc1) = x_t_locs.shape
        # time_steps = np.linspace(0, nof_steps * tau, nof_steps + 1)
        elev0 = 90
        azim0 = 0
        ax.view_init(elev=elev0, azim=azim0)
        scatter_size0 = 20
        max_arrow_len = 10
        arrow_len_mult = max_arrow_len / np.max(np.abs(x_t_vels))
        for target_idx in np.arange(nof_targets):
            # for target_idx in (0,):
            # ax.plot3D(x_t[:,target_idx,0], x_t[:, target_idx,2],time_steps, color=colormap[target_idx%len(colormap)], drawstyle='default')
            if draw_parts:
                ax.scatter(x_t_locs[:, target_idx, 0], x_t_locs[:, target_idx, 1], time_steps, cmap='jet', marker='o', c=time_steps, s=scatter_size0, alpha=1)
            if draw_line:
                ax.plot(x_t_locs[:, target_idx, 0], x_t_locs[:, target_idx, 1], time_steps, color=colormap[target_idx % len(colormap)], drawstyle='default', linewidth=2)
            if draw_arrows:
                for time_idx in time_steps:
                    # for time_idx in (6,):
                    a = Arrow3D([x_t_locs[time_idx, target_idx, 0], x_t_locs[time_idx, target_idx, 0] + arrow_len_mult * x_t_vels[time_idx, target_idx, 0]],
                                [x_t_locs[time_idx, target_idx, 1], x_t_locs[time_idx, target_idx, 1] + arrow_len_mult * x_t_vels[time_idx, target_idx, 1]],
                                [time_idx, time_idx], mutation_scale=20, lw=3, arrowstyle="wedge", color="g", alpha=0.3)
                    ax.add_artist(a)
            # ax.scatter3D(x_t[:, target_idx, 0], x_t[:, target_idx, 2], time_steps, c=time_steps, cmap='jet', s=scatter_size0);
        if do_show:
            plt.draw()
            plt.show(block=False)

    def plot_2d_particle_traj_at_ts(self, x_t_locs, x_t_vels, ts_idx, ax=None, draw_parts=True, draw_arrows=True):
        assert len(x_t_locs.shape) == 3
        do_show = False
        center = self.opt.sensor_params.center
        sensor_size = self.opt.sensor_params.sensor_size
        if ax == None:
            do_show = True
            plt.figure()
            ax = plt.axes()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim(center[0] - sensor_size[0] / 2, center[0] + sensor_size[0] / 2)
            ax.set_ylim(center[1] - sensor_size[1] / 2, center[1] + sensor_size[1] / 2)
            # plt.show(block=False)

        (nof_time_steps, nof_targets, dc1) = x_t_locs.shape
        # time_steps = np.linspace(0, nof_steps * tau, nof_steps + 1)
        time_steps = np.arange(nof_time_steps)
        elev0 = 90
        azim0 = 0
        scatter_size0 = 20
        max_arrow_len = 10
        arrow_len_mult = max_arrow_len / np.max(np.abs(x_t_vels))
        # arrow_len_mult = 1
        for target_idx in np.arange(nof_targets):
            # for target_idx in (0,):
            # ax.plot3D(x_t[:,target_idx,0], x_t[:, target_idx,2],time_steps, color=colormap[target_idx%len(colormap)], drawstyle='default')
            if draw_parts:
                ax.scatter(x_t_locs[ts_idx, target_idx, 0], x_t_locs[ts_idx, target_idx, 1], cmap='jet', marker='o', s=scatter_size0, alpha=1)
            if draw_arrows:
                ax.annotate("",
                            xy=(x_t_locs[ts_idx, target_idx, 0] + arrow_len_mult * x_t_vels[ts_idx, target_idx, 0], x_t_locs[ts_idx, target_idx, 1] + arrow_len_mult * x_t_vels[ts_idx, target_idx, 1]),
                            xytext=(x_t_locs[ts_idx, target_idx, 0], x_t_locs[ts_idx, target_idx, 1]),
                            arrowprops=dict(arrowstyle="->", facecolor='g', edgecolor='g'))
                # ax.arrow(dx = arrow_len_mult * x_t[ts_idx, target_idx, 1],
                #         dy = arrow_len_mult * x_t[ts_idx, target_idx, 3],
                #         x = x_t[ts_idx, target_idx, 0],
                #         y = x_t[ts_idx, target_idx, 2],
                #         width=0.08, facecolor='g',edgecolor='g')
            # ax.scatter3D(x_t[:, target_idx, 0], x_t[:, target_idx, 2], time_steps, c=time_steps, cmap='jet', s=scatter_size0);
        if do_show:
            plt.show(block=False)

    def plot_2d_particles_traj_at_ts(self, time_step, real_traj_locs,real_traj_vels, weights, prts_locs, prts_vels, rcnstd_traj_locs, rcnstd_traj_vels,  ax=None, draw_parts=True, draw_arrows=False):
        assert len(prts_locs.shape) == 4
        do_show = False
        center = self.opt.sensor_params.center
        sensor_size = self.opt.sensor_params.sensor_size
        if ax == None:
            do_show = True
            plt.figure()
            ax = plt.axes()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim(center[0] - sensor_size[0] / 2, center[0] + sensor_size[0] / 2)
            ax.set_ylim(center[1] - sensor_size[1] / 2, center[1] + sensor_size[1] / 2)
            plt.suptitle("timestep: "+str(time_step))
            # plt.show(block=False)

        (nof_steps, nof_parts, nof_targets, dc1) = prts_locs.shape
        # time_steps = np.linspace(0, nof_steps * tau, nof_steps + 1)
        scatter_size0 = 20
        max_arrow_len = 10
        arrow_len_mult = max_arrow_len / np.max(np.abs(prts_vels))
        # arrow_len_mult = 1
        draw_real_target = True
        draw_reconstructed_target = True
        for target_idx in np.arange(nof_targets):
            # for target_idx in (0,):
            # ax.plot3D(x_t[:,target_idx,0], x_t[:, target_idx,2],time_steps, color=colormap[target_idx%len(colormap)], drawstyle='default')

            if draw_parts:
                for part_idx in np.arange(nof_parts):
                    ax.scatter(prts_locs[time_step, part_idx, target_idx, 0], prts_locs[time_step, part_idx, target_idx, 1], c=(weights[time_step, part_idx]), cmap='jet', marker='o', s=200 , alpha=0.05 + 0.95* weights[time_step, part_idx])

            if draw_arrows:
                ax.annotate("",
                            xy=(prts_locs[time_step, part_idx, target_idx, 0] + arrow_len_mult * prts_vels[time_step, part_idx, target_idx, 0], prts_locs[time_step, part_idx, target_idx, 1] + arrow_len_mult * prts_vels[time_step, part_idx, target_idx, 1]),
                            xytext=(prts_locs[time_step, part_idx, target_idx, 0], prts_locs[time_step, part_idx, target_idx, 1]),
                            arrowprops=dict(arrowstyle="->", facecolor='g', edgecolor='g'))
                # ax.arrow(dx = arrow_len_mult * x_t[ts_idx, target_idx, 1],
                #         dy = arrow_len_mult * x_t[ts_idx, target_idx, 3],
                #         x = x_t[ts_idx, target_idx, 0],
                #         y = x_t[ts_idx, target_idx, 2],
                #         width=0.08, facecolor='g',edgecolor='g')
            # ax.scatter3D(x_t[:, target_idx, 0], x_t[:, target_idx, 2], time_steps, c=time_steps, cmap='jet', s=scatter_size0);
            if draw_real_target:
                ax.scatter(real_traj_locs[time_step, target_idx, 0], real_traj_locs[time_step, target_idx, 1], c='green', cmap='jet', marker='x', s=100, alpha=1)
                ax.annotate(str(target_idx), xy=(real_traj_locs[time_step, target_idx, 0], real_traj_locs[time_step, target_idx, 1]),c='yellow')
            if draw_reconstructed_target:
                ax.scatter(rcnstd_traj_locs[time_step, target_idx, 0], rcnstd_traj_locs[time_step, target_idx, 1], c='red', cmap='jet', marker='x', s=100, alpha=1)
                ax.annotate(str(target_idx), xy=(rcnstd_traj_locs[time_step, target_idx, 0], rcnstd_traj_locs[time_step, target_idx, 1]),c='yellow')
        if do_show:
            plt.show(block=False)

    def plot_3d_particle_traj_with_particles_and_real_traj(self, rcnstd_traj_locs, rcnstd_traj_vels, prts_locs, prts_vels, weights, real_traj_locs,real_traj_vels, title=""):
        #set_idx = 1
        #rcnstd_traj_locs = rcnstd_traj_locs_batch[set_idx]
        timesteps_recon = np.arange(len(rcnstd_traj_locs))
        timesteps_real = timesteps_recon[len(rcnstd_traj_locs) - len(real_traj_locs):]
        max_parts_to_paint = 10
        assert len(rcnstd_traj_locs.shape) == 3
        assert len(prts_locs.shape) == 4
        assert len(weights.shape) == 2
        center = self.opt.sensor_params.center
        sensor_size = self.opt.sensor_params.sensor_size
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlim(0, 0.003)
        fig.set_figheight(5)
        fig.set_figwidth(5)
        fig.set_dpi(150)
        plt.title('Eigenvectors '+title)
        plt.tight_layout()
        # ax.axis('scaled')  # this line fits your images to screen
        #0.003
        ax.autoscale(enable=True)
        # plt.figure()
        # ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('time')
        ax.set_xlim(center[0] - sensor_size[0] / 2, center[0] + sensor_size[0] / 2)
        ax.set_ylim(center[1] - sensor_size[1] / 2, center[1] + sensor_size[1] / 2)
        elev0 = 90
        azim0 = 0
        ax.view_init(elev=elev0, azim=azim0)
        self.plot_3d_particle_traj(real_traj_locs,real_traj_vels, timesteps_real, ax, draw_line=False, draw_parts=True, draw_arrows=False)
        self.plot_3d_particle_traj(rcnstd_traj_locs, rcnstd_traj_vels, timesteps_recon, ax, draw_line=True, draw_parts=False, draw_arrows=True)

        (nof_time_steps, nof_targets, dc1) = rcnstd_traj_locs.shape
        # time_steps = np.linspace(0, nof_steps * tau, nof_steps + 1)
        time_steps = np.arange(nof_time_steps)

        scatter_size0 = 5
        paint_particls = False
        nof_times, nof_parts, nof_targs, dc1 = prts_locs.shape
        #weights = np.exp(weights)
        #weights = weights / np.tile(np.reshape(np.sum(weights, axis=1), (nof_times, -1)), (1, nof_parts))

        avg_wt = np.average(weights)
        marker_mult = 100 / avg_wt

        time_steps_to_paint = (0, 1, 2, 3)
        time_steps_to_paint = (0, 1,)
        # time_steps_to_paint = np.arange(nof_times)
        targets_to_paint = (0,)

        if nof_times * nof_parts * nof_targs <= max_parts_to_paint:
            paint_particls = True
        max_arrow_len = 10
        arrow_len_mult = max_arrow_len / np.max(np.abs(prts_vels))
        if 1 or paint_particls:
            # for time_step in np.arange(nof_times):
            for time_step in time_steps_to_paint:
                for part_idx in np.arange(nof_parts):
                    # ax.scatter(prts_locs[time_step, part_idx, :, 0], prts_locs[time_step, part_idx, :, 1], time_step, marker='o', c='k', s=marker_mult * weights[time_step, part_idx], alpha=weights[time_step, part_idx])
                    for targ_idx in targets_to_paint:
                        # for targ_idx in np.arange(nof_targs):
                        a = Arrow3D([prts_locs[time_step, part_idx, targ_idx, 0], prts_locs[time_step, part_idx, targ_idx, 0] + arrow_len_mult * prts_vels[time_step, part_idx, targ_idx, 0]],
                                    [prts_locs[time_step, part_idx, targ_idx, 1], prts_locs[time_step, part_idx, targ_idx, 1] + arrow_len_mult * prts_vels[time_step, part_idx, targ_idx, 1]],
                                    [time_step, time_step], mutation_scale=20, lw=3, arrowstyle="wedge", color="r", alpha=0.1)
                        # ax.add_artist(a)
                        # ax.scatter(prts_locs[time_step, part_idx, targ_idx, 0], prts_locs[time_step, part_idx, targ_idx, 1], time_step, marker='o', c='k', s=10+1000 * weights[time_step, part_idx], alpha=0.1 + 0.9 * weights[time_step, part_idx])
                        ax.scatter(prts_locs[time_step, part_idx, targ_idx, 0], prts_locs[time_step, part_idx, targ_idx, 1], time_step, marker='o', c='k', s=10 + np.maximum(0, (100 * (weights[time_step, part_idx] - avg_wt))), alpha=0.1 + 0.9 * weights[time_step, part_idx])
                self.plot_2d_particles_traj_at_ts(time_step, real_traj_locs,real_traj_vels, weights, prts_locs, prts_vels, rcnstd_traj_locs, rcnstd_traj_vels,  ax=None, draw_parts=True, draw_arrows=False)
        plt.title(title)
        plt.draw()
        plt.show(block=False)


class TrajectoryStep(object):
    """
    Store particle approximation, input, output and timestamp for
    a single time index in a trajectory

    Args:
     - pa (ParticleAppromixation): particle approximation
     - y (array-like): measurements at time t
       (y[t] is the measurment of x[t])
     - t (float): time stamp for time t
     - ancestors (array-like): indices for each particles ancestor
    """
    def __init__(self, pa, ancestors=None):
        self.pa = pa
        self.ancestors = ancestors



class ParticleApproximation(object):
    """
    Contains collection of particles approximating a pdf

    Use either seed and nof_parts or particles (and optionally weights,
    if not uniform)

    Args:
     - particles (array-like): collection of particles
     - weights (array-like): weight for each particle
     - seed (array-like): value to initialize all particles with
     - nof_parts (int): number of particles

    """
    def __init__(self, particles=None, logw=None, seed=None, nof_parts=None):
        if (particles is not None):
            self.part = numpy.copy(numpy.asarray(particles))
            batch_size =  len(particles)
            nof_parts = len(particles[0])
        else:
            self.part = numpy.empty(nof_parts, type(seed))
            for k in range(nof_parts):
                self.part[k] = copy.deepcopy(seed)

        if (logw is not None):
            self.w = numpy.copy(logw)
        else:
            self.w = numpy.tile(numpy.reshape(-math.log(nof_parts) * numpy.ones(nof_parts), (1, nof_parts)), (batch_size, 1))

        self.nof_parts = nof_parts

        # Used to keep track of the offest on all weights, this is continually updated
        # when the weights are rescaled to avoid going to -Inf
        self.w_offset = 0.0*numpy.ones((batch_size))

    def __len__(self):
        return len(self.part)



