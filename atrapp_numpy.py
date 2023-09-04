from models import *



def get_z_for_particles_at_timestep(particles):
    # particle should be of shape = [nof_particles, nof_targets, state_dim]
    pad = pad_mult * interp_sig
    pad_tiles = int(np.ceil(pad / dt))
    assert len(particles.shape) == 3
    nof_particles = particles.shape[0]
    z_coo_x = center[0] - sensor_size[0] / 2 - pad_tiles * dt + np.tile(dt * np.arange(nof_s_x + 2 * pad_tiles).reshape((1, 1, 1, nof_s_x + 2 * pad_tiles)), [nof_particles, nof_targets, nof_s_y + 2 * pad_tiles, 1])
    z_coo_y = center[1] - sensor_size[1] / 2 - pad_tiles * dt + np.tile(dt * np.arange(nof_s_y + 2 * pad_tiles).reshape((1, 1, nof_s_y + 2 * pad_tiles, 1)), [nof_particles, nof_targets, 1, nof_s_x + 2 * pad_tiles])

    particles_xs = np.tile(particles[:, :, 0].reshape((*particles.shape[:-1], 1, 1)), [1, 1, nof_s_y + 2 * pad_tiles, nof_s_x + 2 * pad_tiles])
    particles_ys = np.tile(particles[:, :, 2].reshape((*particles.shape[:-1], 1, 1)), [1, 1, nof_s_y + 2 * pad_tiles, nof_s_x + 2 * pad_tiles])
    if is_gaussian:
        z_snrs = np.sum(np.exp(-0.5 * (np.power(z_coo_x - particles_xs, 2) + np.power(z_coo_y - particles_ys, 2)) / np.power(interp_sig, 2)), axis=1)
    else:
        z_snrs = np.sum(snr0db * snr0db / (eps + np.power(z_coo_x - particles_xs, 2) + np.power(z_coo_y - particles_ys, 2)), axis=1)
        z_snrs = np.minimum(snr0db, z_snrs)
    all_z_sum_check = np.sum(np.sum(z_snrs, axis=-1), axis=-1)
    return z_snrs


def get_lh_measure_particles_with_measurement_numpy(particles, measurement):
    assert len(measurement.shape) == 2
    assert len(particles.shape) == 3

    z_for_particles = get_z_for_particles_at_timestep(particles)

    if 0:
        maxes_z_for_particles = np.max(np.max(z_for_particles, axis=-1), axis=-1)
        z_for_particles_normed = z_for_particles / np.tile(np.reshape(maxes_z_for_particles, (maxes_z_for_particles.shape[0], 1, 1)), [1, *z_for_particles.shape[-2:]])
        z_for_particles = z_for_particles_normed

    pad_vert = int((z_for_particles.shape[-2] - measurement.shape[-2]) / 2)
    pad_hor = int((z_for_particles.shape[-1] - measurement.shape[-1]) / 2)
    measurement_padded = np.pad(measurement, ((pad_vert, pad_vert), (pad_hor, pad_hor)))
    z_for_meas_rep = np.tile(measurement_padded, [z_for_particles.shape[0], 1, 1])
    # if do_threshold_measurements and not is_gaussian:
    #    z_for_meas_rep = np.minimum(z_for_meas_rep,threshold_measurements_th)
    #    z_for_particles = np.minimum(z_for_particles, threshold_measurements_th)
    if limit_sensor_exp:
        z_lh_ln = -np.power(z_for_meas_rep - z_for_particles, 2) / 2 / lh_sig2
        z_lh_ln += -np.max(z_lh_ln)
        indcs = np.argwhere(z_lh_ln < -meas_particle_lh_exp_power_max)
        z_lh_ln[indcs] += (-z_lh_ln[indcs] - meas_particle_lh_exp_power_max) * (1)
        pz_x_log = np.sum(np.sum(z_lh_ln, axis=-1), axis=-1)
        pz_x_log += -np.max(pz_x_log)
        exp_eps = 10
        indcs = np.argwhere(pz_x_log < -exp_eps)
        pz_x_log[indcs] += (-pz_x_log[indcs] - exp_eps) * (1)
        pz_x = np.exp(pz_x_log)
    else:
        z_lh = np.exp(-np.power(z_for_meas_rep - z_for_particles, 2) / 2 / lh_sig2)
        # sigma0 = 0.5
        # z_lh = np.exp(-np.power(gaussian_filter(z_for_meas_rep, sigma=sigma0) - gaussian_filter(z_for_particles, sigma=sigma0), 2) / 2 / sig2)
        # z_lh = np.minimum(1, 1/(eps+ np.power(z_for_meas_rep - z_for_particles,2)))
        pz_x = np.prod(np.prod(z_lh, axis=-1), axis=-1)
        # pz_x = np.exp(np.sum(np.sum(np.log(z_lh), axis=-1), axis=-1))
        # all_z_sum_check = np.sum(np.sum(z_for_particles, axis=-1), axis=-1)
    return pz_x


def advance_samples(is_mu, old_particles, F, Q):
    assert len(old_particles.shape) == 3
    if is_mu:
        noises = np.zeros_like(old_particles)
    else:
        noises = np.random.multivariate_normal(np.zeros((old_particles.shape[-1])), Q, old_particles.shape[:-1])
    new_particles = np.transpose(F @ np.transpose(old_particles, (0, 2, 1)), (0, 2, 1)) + noises
    return new_particles



def update_atrapp_numpy(args):
    old_particles, old_weights, z_for_meas, is_first_sample_mu, is_TRAPP, F, Q = args
    # old_particles.shape = (nof_parts, nof_targs, state_vector_dim)
    assert len(old_particles.shape) == 3

    nof_parts = old_particles.shape[0]
    nof_targs = old_particles.shape[1]
    state_vector_dim = old_particles.shape[2]

    new_particles0 = advance_samples(is_first_sample_mu, old_particles, F, Q)
    final_weights = np.ones((nof_parts,)) / nof_parts
    if z_for_meas is None:
        new_particles = old_particles
        meas_part_of_initial_sampling = np.ones((nof_parts,))
    else:
        assert len(z_for_meas.shape) == 2
        new_particles = np.zeros_like(old_particles)
        bj_x_kp1 = np.zeros(old_particles.shape[:-1])
        weighted_avg_particle = np.transpose(old_weights.reshape((1, -1)) @ np.transpose(new_particles0, (1, 0, 2)), (1, 0, 2))
        X_hat = weighted_avg_particle / np.sum(old_weights)
        X_hat_tiled = np.tile(X_hat, (nof_parts, 1, 1))

        for targ_idx in np.arange(nof_targs):
            # print("on target "+str(targ_idx)) #taking same target on all particles
            curr_target_new_particles0 = new_particles0[:, targ_idx].reshape(nof_parts, 1, state_vector_dim)
            # first weighting start
            targs_indcs = (*np.arange(targ_idx), *np.arange(targ_idx + 1, nof_targs))
            Xmj_hat = X_hat_tiled[:, ((targs_indcs))]
            curr_X_hat = np.concatenate((Xmj_hat, curr_target_new_particles0), axis=1)
            bj_mu = get_lh_measure_particles_with_measurement_numpy(curr_X_hat, z_for_meas)
            new_weights = np.multiply(old_weights, bj_mu)
            if np.isnan(bj_mu).any():
                dsfsfs = 4

            new_weights = new_weights / np.sum(new_weights)  # get_particles_lh_with_real_state(measurement, curr_X_hat, time, sig2)
            # first weighting end (new_weights is lambda on TRAPP alg)
            # resampling
            if 0:
                fig, axs = plt.subplots()
                # plt.sca(axes[1, 1])
                axs.plot(np.sort(new_weights), 'ro')
                plt.show(block=False)

            try:
                sampled_indcs = np.random.choice(np.arange(nof_parts), nof_parts, replace=True, p=new_weights)
            except:
                sdfa = 5
            # print(sampled_indcs)
            # redrawing, advancing-sampling but indices=sampled_indcs
            curr_target_old_particles = old_particles[sampled_indcs][:, targ_idx].reshape(nof_parts, 1, state_vector_dim)
            x_star = advance_samples(False, curr_target_old_particles, F, Q)
            # bj_x_kp1 is the importance density, or sampling distrbution, but for the specific target on the specific particle
            # bj_x_kp1 dpends on measurement and particle location (disregards weights of any kind)
            if not is_TRAPP:
                curr_target_new_particles = x_star
                bj_x_kp1[:, targ_idx] = bj_mu[sampled_indcs]
            else:
                # finiding new weights
                curr_X_hat = np.concatenate((Xmj_hat, x_star), axis=1)
                bj_x_star = get_lh_measure_particles_with_measurement_numpy(curr_X_hat, z_for_meas)
                rj_x_star = bj_x_star / bj_mu[sampled_indcs]
                # normalizing weights
                rj_x_star = rj_x_star / np.sum(rj_x_star)
                # resampling from x_star, according to rj_x_star
                sampled_indcs = np.random.choice(np.arange(nof_parts), nof_parts, replace=True, p=rj_x_star)
                curr_target_new_particles = x_star[sampled_indcs]
                bj_x_kp1[:, targ_idx] = bj_x_star[sampled_indcs]
            new_particles[:, targ_idx] = curr_target_new_particles.reshape(new_particles[:, targ_idx].shape)
        # normalizing bj_x_kp1 so that the multipicazation wont be very small (the pi)
        #
        bj_x_kp1_normed = bj_x_kp1  # / np.max(bj_x_kp1)
        pi_target_bj = np.prod(bj_x_kp1_normed, axis=1)
        meas_part_of_initial_sampling = pi_target_bj  # / np.max(pi_target_bj)
    # q_of_particle has the effective sampling weight for when the particle was sampled,
    # it depended on the particles level of fitness to the measrements at the time of sampling
    return new_particles, final_weights, meas_part_of_initial_sampling
