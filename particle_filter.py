####################################################################
# https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
####################################################################
from atrapp_torch import *
from atrapp_numpy import *


###############################################################################
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


###############################################################################
def get_ospa_distances(x, wted_avg_traj, p=2, c=10):
    nof_times, nof_targs, targ_dim = x.shape
    ospa = ospa_metric.OSPAMetric(p=p, c=c)
    ospa_results = []
    for time in np.arange(nof_times):
        part_state_true = []
        part_state_est = []
        for targ in np.arange(nof_targs):
            part_state_true.append(ospa_metric.State(ss_tps_arr.StateVector(x[time, targ, (0, 2)])))
            part_state_est.append(ospa_metric.State(ss_tps_arr.StateVector(wted_avg_traj[time, targ, (0, 2)])))
        ospa_results.append(ospa.compute_OSPA_distance(part_state_true, part_state_est))
    ospa_dists = np.zeros(nof_times)
    for idx, res in enumerate(ospa_results):
        ospa_dists[idx] = res.value
    return ospa_dists


###################################################################


def get_z_for_states_in_times_through_torch(particles):
    # len(particles.shape) should be 3
    particles = torch.from_numpy(particles).to(device)
    z_snrs = get_z_for_particles_at_timestep_torch(particles)
    pad = pad_mult * interp_sig
    z_snrs = z_snrs[:, int(pad / dt):z_snrs.shape[-2] - int(pad / dt), int(pad / dt):z_snrs.shape[-1] - int(pad / dt)]
    z_snrs = z_snrs.cpu().detach().numpy()
    return z_snrs

def get_lh_measure_particles_with_measurement(particles, measurement,  return_log = False):
    if is_torch_and_not_numpy:
        return get_lh_measure_particles_with_measurement_to_torch(particles, measurement, device, return_log = return_log)
    else:
        return get_lh_measure_particles_with_measurement_numpy(particles, measurement)


# no atrapp should be used with only 1 target
def update_with_model_only(particles, weights, F, Q):
    new_particles = advance_samples(False, particles, F, Q)
    new_wts = weights
    meas_part_of_initial_sampling = np.ones(particles.shape[0])
    return new_particles, new_wts, meas_part_of_initial_sampling


def update_atrapp(args):
    if is_torch_and_not_numpy:
        return update_atrapp_to_torch(args, device=device)
    else:
        return update_atrapp_numpy(args)



#def measure_atrapp(new_particles, weights, z_for_meas, meas_part_of_initial_sampling):
#    new_weights = weights * get_lh_measure_particles_with_measurement_numpy(new_particles, z_for_meas) / meas_part_of_initial_sampling
#    final_weights = new_weights / np.sum(new_weights)
#
#    return new_particles, final_weights


###################################################################
# shouldnt be used in code cause compares particlles and particles instead of partiles and measurements

def get_particles_lh_with_real_state(real_state_all_times, particles_all_times, time, sig2):
    assert len(real_state_all_times.shape) == 4
    assert len(particles_all_times.shape) == 4
    measurement_len = real_state_all_times.shape[0]
    real_state = real_state_all_times[:, time]
    # real_state = real_state.reshape((measurement_len, *real_state.shape))
    # particles_len = particles.shape[0]
    particles = particles_all_times[:, time]
    z_for_particles = get_z_for_particles_at_timestep(particles)
    z_for_meas = get_z_for_particles_at_timestep(real_state)
    z_for_meas_rep = np.tile(z_for_meas, [z_for_particles.shape[0], 1, 1])
    # print(1 / np.sqrt(2 * np.pi * sig2))
    # z_lh = 1 / np.sqrt(2 * np.pi * sig2) * np.exp(-np.power(z_for_meas_rep - z_for_particles, 2) / 2 / sig2)
    z_lh = np.exp(-np.power(z_for_meas_rep - z_for_particles, 2) / 2 / sig2)
    pz_x = np.exp(np.sum(np.sum(np.log(z_lh), axis=-1), axis=-1))
    return pz_x


###############################################################################
def plot_3d_particle_traj(x_t, ax=None, draw_line=True, draw_parts=True, draw_arrows=True):
    assert len(x_t.shape) == 3
    do_show = False
    if ax == None:
        do_show = True
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
    ax.view_init(elev=elev0, azim=azim0)
    scatter_size0 = 20
    max_arrow_len = 10
    arrow_len_mult = max_arrow_len / np.max(np.abs(x_t[:, :, (1, 3)]))
    for target_idx in np.arange(nof_targets):
        # for target_idx in (0,):
        # ax.plot3D(x_t[:,target_idx,0], x_t[:, target_idx,2],time_steps, color=colormap[target_idx%len(colormap)], drawstyle='default')
        if draw_parts:
            ax.scatter(x_t[:, target_idx, 0], x_t[:, target_idx, 2], time_steps, cmap='jet', marker='o', c=time_steps, s=scatter_size0, alpha=1)
        if draw_line:
            ax.plot(x_t[:, target_idx, 0], x_t[:, target_idx, 2], time_steps, color=colormap[target_idx%len(colormap)], drawstyle='default', linewidth=2)
        if draw_arrows:
            for time_idx in np.arange(nof_time_steps):
                # for time_idx in (6,):
                a = Arrow3D([x_t[time_idx, target_idx, 0], x_t[time_idx, target_idx, 0] + arrow_len_mult * x_t[time_idx, target_idx, 1]],
                            [x_t[time_idx, target_idx, 2], x_t[time_idx, target_idx, 2] + arrow_len_mult * x_t[time_idx, target_idx, 3]],
                            [time_idx, time_idx], mutation_scale=20, lw=3, arrowstyle="wedge", color="g", alpha=0.3)
                ax.add_artist(a)
        # ax.scatter3D(x_t[:, target_idx, 0], x_t[:, target_idx, 2], time_steps, c=time_steps, cmap='jet', s=scatter_size0);
    if do_show:
        plt.draw()
        plt.show(block=False)

def plot_2d_particle_traj_at_ts(x_t, ts_idx, ax=None, draw_parts=True, draw_arrows=True):
    assert len(x_t.shape) == 3
    do_show = False
    if ax == None:
        do_show = True
        plt.figure()
        ax = plt.axes()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(center[0] - sensor_size[0] / 2, center[0] + sensor_size[0] / 2)
        ax.set_ylim(center[1] - sensor_size[1] / 2, center[1] + sensor_size[1] / 2)
        #plt.show(block=False)

    (nof_time_steps, nof_targets, dc1) = x_t.shape
    # time_steps = np.linspace(0, nof_steps * tau, nof_steps + 1)
    time_steps = np.arange(nof_time_steps)
    elev0 = 90
    azim0 = 0
    scatter_size0 = 20
    max_arrow_len = 10
    arrow_len_mult = max_arrow_len / np.max(np.abs(x_t[:, :, (1, 3)]))
    #arrow_len_mult = 1
    for target_idx in np.arange(nof_targets):
        # for target_idx in (0,):
        # ax.plot3D(x_t[:,target_idx,0], x_t[:, target_idx,2],time_steps, color=colormap[target_idx%len(colormap)], drawstyle='default')
        if draw_parts:
            ax.scatter(x_t[ts_idx, target_idx, 0], x_t[ts_idx, target_idx, 2], cmap='jet', marker='o', s=scatter_size0, alpha=1)
        if draw_arrows:
            ax.annotate("",
                        xy=(x_t[ts_idx, target_idx, 0] + arrow_len_mult * x_t[ts_idx, target_idx, 1], x_t[ts_idx, target_idx, 2] + arrow_len_mult * x_t[ts_idx, target_idx, 3]),
                        xytext=(x_t[ts_idx, target_idx, 0], x_t[ts_idx, target_idx, 2]),
                        arrowprops=dict(arrowstyle="->",facecolor='g',edgecolor='g'))
            #ax.arrow(dx = arrow_len_mult * x_t[ts_idx, target_idx, 1],
            #         dy = arrow_len_mult * x_t[ts_idx, target_idx, 3],
            #         x = x_t[ts_idx, target_idx, 0],
            #         y = x_t[ts_idx, target_idx, 2],
            #         width=0.08, facecolor='g',edgecolor='g')
        # ax.scatter3D(x_t[:, target_idx, 0], x_t[:, target_idx, 2], time_steps, c=time_steps, cmap='jet', s=scatter_size0);
    if do_show:
        plt.show(block=False)

def plot_3d_particle_traj_with_particles_and_real_traj(rcnstd_traj, particles, weights, real_traj):
    assert len(rcnstd_traj.shape) == 3
    assert len(particles.shape) == 4
    assert len(weights.shape) == 2

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Eigenvectors')
    plt.tight_layout()
    #ax.axis('scaled')  # this line fits your images to screen
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
    plot_3d_particle_traj(real_traj, ax, draw_line=False, draw_parts=True, draw_arrows=False)
    plot_3d_particle_traj(rcnstd_traj, ax, draw_line=True, draw_parts=False, draw_arrows=True)

    (nof_time_steps, nof_targets, dc1) = rcnstd_traj.shape
    # time_steps = np.linspace(0, nof_steps * tau, nof_steps + 1)
    time_steps = np.arange(nof_time_steps)

    scatter_size0 = 5
    paint_particls = False
    nof_times, nof_parts, nof_targs, dc1 = particles.shape

    avg_wt = np.average(weights)
    # max_wt = np.max(weights[time_step])
    marker_mult = 100 / avg_wt

    time_steps_to_paint = (1,2,3,4, 5, 6, 20)
    time_steps_to_paint = (8,)
    #time_steps_to_paint = np.arange(nof_times)
    targets_to_paint = (0,)

    if nof_times * nof_parts * nof_targs <= max_parts_to_paint:
        paint_particls = True
    max_arrow_len = 10
    arrow_len_mult = max_arrow_len / np.max(np.abs(particles[:, :, :, (1, 3)]))
    if 1 or paint_particls:
         #for time_step in np.arange(nof_times):
         for time_step in time_steps_to_paint:
            for part_idx in np.arange(nof_parts):
                #ax.scatter(particles[time_step, part_idx, :, 0], particles[time_step, part_idx, :, 2], time_step, marker='o', c='k', s=marker_mult * weights[time_step, part_idx], alpha=weights[time_step, part_idx])
                for targ_idx in targets_to_paint:
                    # for targ_idx in np.arange(nof_targs):
                    a = Arrow3D([particles[time_step, part_idx, targ_idx, 0], particles[time_step, part_idx, targ_idx, 0] + arrow_len_mult * particles[time_step, part_idx, targ_idx, 1]],
                                [particles[time_step, part_idx, targ_idx, 2], particles[time_step, part_idx, targ_idx, 2] + arrow_len_mult * particles[time_step, part_idx, targ_idx, 3]],
                                [time_step, time_step], mutation_scale=20, lw=3, arrowstyle="wedge", color="r", alpha=0.1)
                    #ax.add_artist(a)
                    ax.scatter(particles[time_step, part_idx, targ_idx, 0], particles[time_step, part_idx, targ_idx, 2], time_step, marker='o', c='k', s=10, alpha=0.1+0.9*weights[time_step, part_idx])

    plt.draw()
    plt.show(block=False)


def plot_2d_particles_and_real_traj_at_ts(rcnstd_traj, particles, weights, real_traj, true_vels, ts_idx):
    assert len(rcnstd_traj.shape) == 3
    assert len(particles.shape) == 4
    assert len(weights.shape) == 2

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot()
    plt.title('Eigenvectors')
    plt.tight_layout()
    #ax.axis('scaled')  # this line fits your images to screen
    ax.autoscale(enable=True)
    # plt.figure()
    # ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(center[0] - sensor_size[0] / 2, center[0] + sensor_size[0] / 2)
    ax.set_ylim(center[1] - sensor_size[1] / 2, center[1] + sensor_size[1] / 2)

    plot_2d_particle_traj_at_ts(real_traj,  ts_idx, ax, draw_parts=True, draw_arrows=False)
    plot_2d_particle_traj_at_ts(rcnstd_traj, ts_idx, ax, draw_parts=False, draw_arrows=True)

    (nof_time_steps, nof_targets, dc1) = rcnstd_traj.shape
    # time_steps = np.linspace(0, nof_steps * tau, nof_steps + 1)
    time_steps = np.arange(nof_time_steps)

    scatter_size0 = 5
    paint_particls = False
    nof_times, nof_parts, nof_targs, dc1 = particles.shape

    avg_wt = np.average(weights)
    # max_wt = np.max(weights[time_step])
    marker_mult = 100 / avg_wt

    time_steps_to_paint = (1,2,3,4, 5, 6, 20)
    time_steps_to_paint = (0,1,2,3,4)
    #time_steps_to_paint = np.arange(nof_times)
    targets_to_paint = (0,)

    if nof_times * nof_parts * nof_targs <= max_parts_to_paint:
        paint_particls = True
    max_arrow_len = 10
    arrow_len_mult = max_arrow_len / np.max(np.abs(particles[:, :, :, (1, 3)]))
    for targ_idx in targets_to_paint:
        ax.annotate("",
                    xy=(real_traj[ts_idx, targ_idx, 0] + arrow_len_mult * real_traj[ts_idx, targ_idx, 1], real_traj[ts_idx, targ_idx, 2] + arrow_len_mult * real_traj[ts_idx, targ_idx, 3]),
                    xytext=(real_traj[ts_idx, targ_idx, 0], real_traj[ts_idx, targ_idx, 2]),
                    arrowprops=dict(arrowstyle="->", facecolor='b', edgecolor='b')
                    )
    if 1 or paint_particls:
         #for time_step in np.arange(nof_times):
        for part_idx in np.arange(nof_parts):

            #ax.scatter(particles[time_step, part_idx, :, 0], particles[time_step, part_idx, :, 2], time_step, marker='o', c='k', s=marker_mult * weights[time_step, part_idx], alpha=weights[time_step, part_idx])
            for targ_idx in targets_to_paint:
                # for targ_idx in np.arange(nof_targs):
                #ax.annotate("",
                #            xy=( particles[ts_idx, part_idx, targ_idx, 0] + arrow_len_mult * particles[ts_idx, part_idx, targ_idx, 1], particles[ts_idx, part_idx, targ_idx, 2] + arrow_len_mult * particles[ts_idx, part_idx, targ_idx, 3]),
                #            xytext=(particles[ts_idx, part_idx, targ_idx, 0], particles[ts_idx, part_idx, targ_idx, 2]),
                #            arrowprops=dict(arrowstyle="->",facecolor='r',edgecolor='r')
                #            )

                ax.scatter(particles[ts_idx, part_idx, targ_idx, 0], particles[ts_idx, part_idx, targ_idx, 2], marker='o', c='k', s=50, alpha=0.1+0.9*weights[ts_idx, part_idx])
                    # ax.scatter(x_t[ts_idx, target_idx, 0], x_t[ts_idx, target_idx, 2], cmap='jet', marker='o', s=scatter_size0, alpha=1)

    plt.draw()
    plt.show(block=False)


def paint_meas_and_parts_induced_pdf(real_state_z, particles, nof_parts_to_paint, real_state=None):
    assert len(real_state_z.shape) == 2
    assert len(particles.shape) == 4
    nof_parts_to_paint = np.minimum(np.minimum(21, particles.shape[0]), nof_parts_to_paint)
    # nof_parts, nof_times, nof_targs, state_vector_dim = particles.shape
    # parts_to_paint = np.reshape(particles[:nof_parts_to_paint,time],(nof_parts_to_paint, nof_targs, state_vector_dim))
    pz_x = get_lh_measure_particles_with_measurement_numpy(particles[:nof_parts_to_paint, time], real_state_z)
    order = np.argsort(-pz_x)
    particles_z = get_z_for_particle_in_loop(particles[:nof_parts_to_paint, time])

    fig, axs = plt.subplots()
    axs.imshow(real_state_z)
    axs.set_title("real_state_z")
    plt.setp(axs, xticks=range(nof_s_x), xticklabels=center[0] - sensor_size[0] / 2 + dt * np.arange(nof_s_x),
             yticks=range(nof_s_y), yticklabels=center[1] - sensor_size[1] / 2 + dt * np.arange(nof_s_y))
    plt.xticks(range(nof_s_x), center[0] - sensor_size[0] / 2 + dt * np.arange(nof_s_x), color='k')
    plt.yticks(range(nof_s_y), center[1] - sensor_size[1] / 2 + dt * np.arange(nof_s_y), color='k')
    plt.show(block=False)

    img_rows = 3
    img_cols = 7
    fig, axs = plt.subplots(img_rows, img_cols)
    plt.setp(axs, xticks=range(nof_s_x), xticklabels=center[0] - sensor_size[0] / 2 + dt * np.arange(nof_s_x),
             yticks=range(nof_s_y), yticklabels=center[1] - sensor_size[1] / 2 + dt * np.arange(nof_s_y))
    plt.xticks(range(nof_s_x), center[0] - sensor_size[0] / 2 + dt * np.arange(nof_s_x), color='k')
    plt.yticks(range(nof_s_y), center[1] - sensor_size[1] / 2 + dt * np.arange(nof_s_y), color='k')
    sum_pz_x = np.sum(pz_x)
    pz_x = pz_x[order]
    particles_z = particles_z[order]
    for j in np.arange(img_rows):
        for i in np.arange(img_cols):
            if j * img_rows + i >= nof_parts_to_paint: break
            axs[j, i].imshow(particles_z[j * img_rows + i])
            axs[j, i].set_title("particles_z " + str(j * img_rows + i) + ", pz_x: %.5f\nprob: %.3f" % (pz_x[j * img_rows + i], pz_x[j * img_rows + i] / sum_pz_x))
#            for targ in np.arange(particles.shape[-2]):
#                axs[j, i].scatter(particles[j * img_rows + i, time][0,0], particles[j * img_rows + i, time][0,2], marker='x', c='r')
#            axs[j, i].set(xlim=(center[0] - sensor_size[0] / 2,  center[0] + sensor_size[0] / 2), ylim=(center[1] - sensor_size[1] / 2,  center[1] + sensor_size[1] / 2))
            # if real_state is not None:
            #    axs[j, i].plot(real_state[0], real_state[2],'bo')

    plt.show(block=False)
    dfsdf = 3


#############################################################################

do_run_compare_particles2 = True
disable_all = False

# if not disable_all and do_run_compare_particles2:
if 0:  # not disable_all and do_run_compare_particles2:

    time = 0

    real_state = x_ts[0]
    real_state = real_state.reshape((1, *real_state.shape))
    real_state_z = get_z_for_particle_in_loop(real_state[:, time])
    # real_state_z2 = get_z_for_particles_at_timestep(real_state[:, time])
    # real_state_z=real_state_z2
    real_state_z = real_state_z.reshape((real_state_z.shape[-2], real_state_z.shape[-1]))

    nof_parts_to_paint = 30
    nof_parts_to_paint = np.minimum(nof_parts_to_paint, len(x_ts))

    particles = x_ts[0:nof_parts_to_paint]
    pz_x = get_lh_measure_particles_with_measurement_numpy(particles[:, time], real_state_z)

    paint_meas_and_parts_induced_pdf(real_state_z, particles, nof_parts_to_paint, real_state[:, time].reshape(-1))
    wrwqe = 5
    if 0:  # for inllustration:
        particles_z = get_z_for_particle_in_loop(particles[:, time])
        # particles_z2 = get_lh_measure_particles_with_measurement_numpy(particles[:,time])
        # particles_z - particles_z2

        fig, axs = plt.subplots(2, 2)
        # plt.sca(axes[1, 1])
        plt.setp(axs, xticks=range(nof_s_x), xticklabels=center[0] - sensor_size[0] / 2 + dt * np.arange(nof_s_x),
                 yticks=range(nof_s_y), yticklabels=center[1] - sensor_size[1] / 2 + dt * np.arange(nof_s_y))
        plt.xticks(range(nof_s_x), center[0] - sensor_size[0] / 2 + dt * np.arange(nof_s_x), color='k')
        plt.yticks(range(nof_s_y), center[1] - sensor_size[1] / 2 + dt * np.arange(nof_s_y), color='k')
        axs[0, 0].imshow(real_state_z)
        axs[0, 0].set_title("real_state_z")
        axs[0, 1].imshow(particles_z[0])
        axs[0, 1].set_title("particles_z 0, pz_x: " + str(pz_x[0]))
        axs[1, 0].imshow(particles_z[1])
        axs[1, 0].set_title("particles_z 1, pz_x: " + str(pz_x[1]))
        axs[1, 1].imshow(particles_z[2])
        axs[1, 1].set_title("particles_z 2, pz_x: " + str(pz_x[2]))
        plt.show(block=False)

if not disable_all and 0:
    time = 0
    old_particles = x_ts[:, time]
    nof_parts = old_particles.shape[0]
    old_weights = np.random.uniform(0, 1, nof_parts)
    old_weights = old_weights / np.sum(old_weights)

    measurement_particle = x_ts[0, 0].reshape(1, 8, 4)
    z_for_meas = get_z_for_particles_at_timestep(measurement_particle).reshape(nof_s_y, nof_s_x)

    is_first_sample_mu = False
    is_TRAPP = False

    args = old_particles, old_weights, z_for_meas, is_first_sample_mu, is_TRAPP, F, Q
    new_particles, weights, meas_part_of_initial_sampling = update_atrapp_numpy(args)

    final_particles, final_weights = measure_atrapp(new_particles, weights, z_for_meas, meas_part_of_initial_sampling)

sfwfwq = 8
