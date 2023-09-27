import copy
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from BatchData import *
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
        self.train_nn3 = True

    def clear_db(self, x_with_t0):
        batch_size, nof_steps, nof_targs, _ = x_with_t0.shape
        self.lost_targ_dist = torch.tensor(self.opt.lost_targ_dist, device=self.device)
        self.debug_sav_all_parts = False
        if self.opt.inference_mode == 'paint':
            self.debug_sav_all_parts = True
            self.full_traj_parts_bd = BatchData(batch_size, self.opt.nof_steps+1, self.opt.nof_parts, 1, device='cpu')

    def get_sensor_frame_of_idxinB_ts_targ(self, idx_in_batch, curr_ts, targ_idx_to_zoom, sensor_frame):
        ymins, ymaxs, xmins, xmaxs = sensor_frame
        ymin = ymins[idx_in_batch, curr_ts, targ_idx_to_zoom]
        ymax = ymaxs[idx_in_batch, curr_ts, targ_idx_to_zoom]
        xmin = xmins[idx_in_batch, curr_ts, targ_idx_to_zoom]
        xmax = xmaxs[idx_in_batch, curr_ts, targ_idx_to_zoom]
        return ymin, ymax, xmin, xmax

    def get_XY_grid(self, curr_sensor_frame, margin, pix_per_meter, device):
        ymin, ymax, xmin, xmax = curr_sensor_frame  # self.get_sensor_frame_of_idxinB_ts_targ(idx_in_batch, curr_ts, targ_idx_to_zoom, curr_sensor_frame)
        per_dim_pixels = 2 * margin * torch.tensor(pix_per_meter, device=device)
        per_dim_pixels_int = int(per_dim_pixels.detach().cpu().numpy())
        Y_for_paint = torch.linspace(ymin.data, ymax.data, per_dim_pixels_int, device=device)
        X_for_paint = torch.linspace(xmin.data, xmax.data, per_dim_pixels_int, device=device)
        X_for_paint, Y_for_paint = torch.meshgrid((X_for_paint, Y_for_paint), indexing='xy')
        return X_for_paint, Y_for_paint

    def get_ospa_dist(self, true_x, ess_x, p, c, use_ratio=False):
        batch_size, nof_steps, nof_targs, _ = true_x.shape
        dists = torch.sqrt(
            torch.pow(true_x[:, :, :, 0] - ess_x[:, :, :, 0], 2) +
            torch.pow(true_x[:, :, :, 2] - ess_x[:, :, :, 1], 2)
        )
        minimum_dist_c = torch.where(dists <= c, dists, c)
        ospa_batch = torch.pow(
            1 / nof_targs * torch.sum(
                torch.pow(minimum_dist_c, p
                          ) + 0, dim=2
            ), 1 / p
        )
        return ospa_batch

    def get_dist_sqd(self, true_x, ess_x, p, c):
        batch_size, nof_steps, nof_targs, _ = true_x.shape
        dists = torch.pow(true_x[:, :, :, 0] - ess_x[:, :, :, 0], 2) + torch.pow(true_x[:, :, :, 2] - ess_x[:, :, :, 2], 2)
        return dists

    def get_lost_targs_mask(self, trajs_to_advance_orig, x_with_t0, ts_idx_to_start, nof_steps_to_run, all_ts_avg_prts_locs_mapped, batch_size, nof_targs):
        temp_lost_targs = (torch.sum(torch.pow(x_with_t0[:, ts_idx_to_start + nof_steps_to_run - 1, :, (0, 2)] - all_ts_avg_prts_locs_mapped[:, -1], 2), dim=-1) >= torch.pow(self.lost_targ_dist, 2))
        lost_targs_mask = torch.zeros((batch_size, nof_targs), dtype=torch.bool, device=temp_lost_targs.device)
        ffff = torch.nonzero(temp_lost_targs)
        ffff[:, 0] = trajs_to_advance_orig[ffff[:, 0]]
        lost_targs_mask[torch.split(ffff, 1, dim=1)] = True
        if 0: print("lost_targs_mask: " + str(torch.transpose(lost_targs_mask, 0, 1)))
        return lost_targs_mask

    def get_one_ts_loss(self, x_with_t0, ts_idx, relevant_trajs, temp_bd, curr_measmnts, true_vels, true_locs, batch_size, nof_targs):
        timings = self.forward_one_time_step(temp_bd, measmnts=curr_measmnts, ts_idx=ts_idx, true_vels=true_vels, true_locs=true_locs)
        if self.debug_sav_all_parts:
            tmp_ln_weights, tmp_prts_locs, tmp_prts_vels, tmp_parents_incs = self.curr_bd.get_batch_data(ts_idx=0)
            self.full_traj_parts_bd.sav_batch_data(ts_idx + 1, tmp_ln_weights.detach().cpu(), tmp_prts_locs.detach().cpu(), tmp_prts_vels.detach().cpu(), tmp_parents_incs.detach().cpu())
        all_ts_avg_prts_locs, all_ts_avg_prts_vels = temp_bd.get_all_ts_avg_particles_and_mapping()
        x_for_ospa_loss = x_with_t0[:, ts_idx:ts_idx + 1]
        ospa_loss_b_ts = self.get_ospa_dist(x_for_ospa_loss, all_ts_avg_prts_locs, self.opt.ospa_p, self.opt.ospa_c)
        loss_b_curr_ts = ospa_loss_b_ts
        ospa_batch_curr_ts = self.get_ospa_dist(copy.copy(x_with_t0[:, ts_idx:ts_idx + 1].detach()), all_ts_avg_prts_locs.detach(), self.opt.ospa_p, 10.)
        lost_targs_mask = self.get_lost_targs_mask(relevant_trajs, x_with_t0, ts_idx, 1, all_ts_avg_prts_locs, batch_size, nof_targs)
        if 0: print("lost_targs_mask: " + str(torch.transpose(lost_targs_mask, 0, 1)))
        loss_b_curr_ts = loss_b_curr_ts
        return loss_b_curr_ts, ospa_batch_curr_ts, lost_targs_mask, timings

    def get_batch_loss(self,x, z):
        batch_size, nof_steps, nof_targs, _ = x.shape
        nof_steps_to_run = x.shape[1]
        trajs_to_advance = torch.arange(batch_size, device=self.device)
        loss_batch_ts = torch.zeros((trajs_to_advance.shape[0], nof_steps_to_run), device=self.device)
        loss_batch_ts_list = []
        ospa_batch_ts = torch.zeros((trajs_to_advance.shape[0], nof_steps_to_run), device=self.device)
        ospa_batch_ts.requires_grad = False

        # output || time |  time    |   real   | zi ([]=db idx) | pi (parents) |part(start->end)| w(start->end)  | traj  |
        # idx (i)|| idx  |start->end|   state  | final on idx=i |              |final on idx=i  | final on idx=i |  len  |
        #===================================================================================================================== |
        #    0   ||  0   |  -1->0   |  None    | z0=None        |  p0= None    | None->random   |   None->w0=1   |  0->1 |
        #    1   ||  1   |   0->1   |  x_db[0] | z1=z_db[0]     |  p1= part0   | part0->part1   |   w0->w1       |  1->2 |
        #    2   ||  2   |   1->2   |  x_db[1] | z2=z_db[1]     |  p2= part1   | part1->part2   |   w1->w2       |  2->3 |
        #    3   ||  3   |   2->3   |  x_db[2] | z2=z_db[2]     |  p3= part2   | part2->part3   |   w2->w3       |  3->4 |

        self.true_vels, self.true_locs, x0, v0 = self.get_cheats(x)
        atrapp_time = 0
        nn3_time = 0
        meas_time = 0
        tss_to_run = np.arange(0, nof_steps_to_run)
        ##################### for heatmap paint start ######################
        self.paint_vars = None
        ##################### for heatmap paint start ######################
        real_ts = False
        self.clear_db(x)
        self.model.reset_before_batch(self.train_nn3, x)
        with torch.no_grad():
            self.curr_bd = self.forward_one_time_step_time_0(batch_size, nof_targs, x0=x0, v0=v0)
        if self.debug_sav_all_parts:
            ln_weights, prts_locs, prts_vels, parents_incs = self.curr_bd.get_batch_data(ts_idx=0)
            self.full_traj_parts_bd.sav_batch_data(0 , ln_weights.detach().cpu(), prts_locs.detach().cpu(), prts_vels.detach().cpu(), parents_incs.detach().cpu())
        ts_idx_idx = -1
        lost_targs_mask = torch.zeros((batch_size,nof_targs), dtype=torch.bool, device=self.device)
        actual_grad_batch_size = 0
        for ts_idx in tss_to_run:
            ts_idx_idx +=1
            idcs_of_indices_of_relevant = torch.randperm(trajs_to_advance.shape[0], device=self.device)
            trajs_to_advance_perrmuted = trajs_to_advance[idcs_of_indices_of_relevant]
            b_idcs_to_grad = trajs_to_advance
            b_idcs_of_idcs_to_grad = torch.arange(len(trajs_to_advance), device=self.device)
            b_idcs_not_to_grad= torch.clone(trajs_to_advance)
            for i in b_idcs_to_grad:
                b_idcs_not_to_grad = b_idcs_not_to_grad[b_idcs_not_to_grad!=i]
            b_idcs_of_idcs_not_to_grad = torch.arange(trajs_to_advance.shape[0], device=self.device)
            for i in b_idcs_of_idcs_to_grad:
                b_idcs_of_idcs_not_to_grad = b_idcs_of_idcs_not_to_grad[b_idcs_of_idcs_not_to_grad!=i]
            x_with_t0 = x[b_idcs_to_grad]
            curr_measmnts = z[:, ts_idx]
            temp_bd_grad = self.curr_bd.get_trajs_tensors(b_idcs_to_grad)
            temp_bd_no_grad = self.curr_bd.get_trajs_tensors(b_idcs_not_to_grad)
            true_vels, true_locs = self.true_vels[b_idcs_to_grad], self.true_locs[b_idcs_to_grad]
            loss_b_curr_ts, ospa_batch_curr_ts, curr_lost_targs_mask, timings = self.get_one_ts_loss(
                x[b_idcs_to_grad], ts_idx, b_idcs_to_grad, temp_bd_grad, curr_measmnts[b_idcs_to_grad],
                self.true_vels[b_idcs_to_grad], self.true_locs[b_idcs_to_grad], batch_size, nof_targs)
            temp_bd_grad.detach_all()
            lost_targs_mask = torch.logical_or(lost_targs_mask, curr_lost_targs_mask)
            self.curr_bd.set_trajs_tensors(temp_bd_grad, b_idcs_to_grad)
            if 0: print("ts after curr_bd: "+str(ts_idx)+", lost_targs_mask: " + str(torch.transpose(lost_targs_mask, 0, 1)))
            if len(b_idcs_not_to_grad) != 0:
                loss_b_curr_ts_dc, ospa_batch_curr_ts_wo_grad, curr_lost_targs_mask, timings_dc = self.get_one_ts_loss(
                    x[b_idcs_not_to_grad], ts_idx, b_idcs_not_to_grad, temp_bd_no_grad, curr_measmnts[b_idcs_not_to_grad], False,
                    self.true_vels[b_idcs_not_to_grad], self.true_locs[b_idcs_not_to_grad], batch_size, nof_targs)
                lost_targs_mask = torch.logical_or(lost_targs_mask, curr_lost_targs_mask)
                self.curr_bd.set_trajs_tensors(temp_bd_no_grad, b_idcs_not_to_grad)
            curr_atrapp_time, curr_nn3_time, curr_meas_time = timings
            atrapp_time += curr_atrapp_time
            nn3_time += curr_nn3_time
            meas_time += curr_meas_time
            loss_batch_ts[b_idcs_of_idcs_to_grad, ts_idx_idx:ts_idx_idx + 1] = loss_b_curr_ts
            ospa_batch_ts[b_idcs_of_idcs_to_grad, ts_idx_idx:ts_idx_idx + 1] = ospa_batch_curr_ts
            if len(b_idcs_of_idcs_not_to_grad)>0:
                ospa_batch_ts[b_idcs_of_idcs_not_to_grad, ts_idx_idx:ts_idx_idx + 1] = ospa_batch_curr_ts_wo_grad
            if ts_idx_idx == len(tss_to_run)-1:
                actual_grad_batch_size += len(b_idcs_of_idcs_to_grad)
        return loss_batch_ts, ospa_batch_ts, lost_targs_mask, (atrapp_time, nn3_time, meas_time), actual_grad_batch_size

    def forward_one_time_step_time_0(self, batch_size, nof_targs, x0, v0):
        curr_bd = BatchData(batch_size, 1, self.opt.nof_parts, nof_targs, self.device)
        curr_bd.detach_all()
        particles0 = torch.from_numpy(self.model.create_initial_estimate(x0, v0, self.model.opt.nof_parts))
        batch_size, nof_parts, nof_targs, state_vector_dim = particles0.shape
        prts_locs = particles0[:, :, :, (0, 2)].to(self.model.device)
        prts_vels = particles0[:, :, :, (1, 3)].to(self.model.device)
        ln_weights = torch.log(torch.ones((batch_size, nof_parts), device=self.model.device) / nof_parts)
        parents_incs = torch.tile(torch.reshape(torch.arange(nof_parts), (1, nof_parts, 1)), (batch_size, 1, nof_targs))
        # old_bd is empty updating only final outputs for bf_ref
        t0_nn1_out_lnw = torch.tile(torch.unsqueeze(ln_weights, -1), (1, 1, nof_targs))
        t0_nn3_out_wts_var = torch.var(torch.softmax(ln_weights.detach(), dim=1), unbiased=False, dim=1)
        curr_bd.sav_batch_data(0, ln_weights, prts_locs, prts_vels, parents_incs)
        return curr_bd


    def forward_one_time_step(self, bd, measmnts, ts_idx, true_vels, true_locs):
        assert measmnts is not None, "only supports this scenrios"
        ln_weights, prts_locs, prts_vels,  parents_incs = bd.get_batch_data(0)
        prts_locs, prts_vels, ln_weights, parents_incs, timings = self.model.forward(
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
        bd.sav_batch_data(0, ln_weights, prts_locs, prts_vels, parents_incs)
        return timings

    def get_cheats(self, x_for_cheats):
        if x_for_cheats.shape[1]>1:
            true_vels = (x_for_cheats[:, 1:, :, (0, 2)] - x_for_cheats[:, :-1, :, (0, 2)]) / self.opt.tau
        else:
            true_vels = torch.zeros_like(x_for_cheats[:, 0:1, :, (0, 2)])
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
        elev0 = 90
        azim0 = 0
        ax.view_init(elev=elev0, azim=azim0)
        scatter_size0 = 20
        max_arrow_len = 10
        arrow_len_mult = max_arrow_len / np.max(np.abs(x_t_vels))
        for target_idx in np.arange(nof_targets):
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
        if do_show:
            plt.draw()
            plt.show(block=False)

    def plot_3d_particle_traj_with_particles_and_real_traj(self, x_wt0_torch, set_idx, title="", ax = None):
        weights, prts_locs, prts_vels, parents_incs = self.full_traj_parts_bd.get_batch_data(ts_idx=None)
        weights = torch.softmax(weights, dim=2)
        weights, prts_locs, prts_vels, parents_incs = weights[set_idx].cpu().detach().numpy(), prts_locs[set_idx].cpu().detach().numpy(), prts_vels[set_idx].cpu().detach().numpy(), parents_incs[set_idx].cpu().detach().numpy()
        all_ts_avg_prts_locs, all_ts_avg_prts_vels = self.full_traj_parts_bd.get_all_ts_avg_particles_and_mapping()
        rcnstd_traj_locs, rcnstd_traj_vels = all_ts_avg_prts_locs[set_idx].cpu().detach().numpy(), all_ts_avg_prts_vels[set_idx].cpu().detach().numpy()
        real_traj_locs = x_wt0_torch[set_idx, :, :, (0, 2)].cpu().detach().numpy()
        real_traj_vels = x_wt0_torch[set_idx, :, :, (1, 3)].cpu().detach().numpy()
        timesteps_recon = np.arange(len(rcnstd_traj_locs))
        timesteps_real = timesteps_recon[len(rcnstd_traj_locs) - len(real_traj_locs):]
        assert len(rcnstd_traj_locs.shape) == 3
        assert len(prts_locs.shape) == 4
        assert len(weights.shape) == 2
        center = self.opt.sensor_params.center
        sensor_size = self.opt.sensor_params.sensor_size
        ax_was_none = False
        if ax==None:
            ax_was_none = True
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_zlim(0, 0.003)
            fig.set_figheight(5)
            fig.set_figwidth(5)
            fig.set_dpi(150)
            plt.title('Eigenvectors '+title)
            plt.tight_layout()
            ax.autoscale(enable=True)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('time')
            ax.set_xlim(center[0] - sensor_size[0] / 2, center[0] + sensor_size[0] / 2)
            ax.set_ylim(center[1] - sensor_size[1] / 2, center[1] + sensor_size[1] / 2)
            elev0 = 90
            azim0 = 0
            ax.view_init(elev=elev0, azim=azim0)
        self.plot_3d_particle_traj(real_traj_locs,real_traj_vels, timesteps_real, ax, draw_line=False, draw_parts=True, draw_arrows=False)
        self.plot_3d_particle_traj(rcnstd_traj_locs, rcnstd_traj_vels, timesteps_recon, ax, draw_line=True, draw_parts=False, draw_arrows=False)
        plt.draw()
        if ax_was_none:
            plt.title(title)
            plt.show(block=False)