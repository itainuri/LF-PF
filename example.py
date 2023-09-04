
import matplotlib.pyplot as plt
import numpy as np
import torch
import models as ms
import particle_filter as atrapp
import time
from AtrappModel import AtrappModel
from PfBatchHandler import PfBatchHandler


def print0(new_particles, new_weights,to_be_advenced_particles_ancestors, meas_part_of_initial_sampling, parents_indces):
    print("new_particles: "+str(new_particles))
    print("new_weights: "+str(new_weights))
    print("to_be_advenced_particles_ancestors: "+str(to_be_advenced_particles_ancestors))
    print("meas_part_of_initial_sampling: "+str(meas_part_of_initial_sampling))
    print("parents_indces: "+str(parents_indces))


##########################################################################
def generate_dataset(full_file_path, nof_targs, nof_steps):
    # prepares the true full trajectory of all targets
    max_nof_targs = 10
    with open(full_file_path, 'rb') as f:
        x_ts = np.load(f)
        f.close()
    assert nof_steps <= x_ts.shape[1]
    assert nof_targs <= x_ts.shape[0]
    idcs = np.random.choice(len(x_ts), max_nof_targs, replace=False)
    x_k = np.transpose(x_ts[idcs], (1, 0, 2, 3)).squeeze(-2)
    x_k_for_run = x_k[:, :nof_targs]
    z_k = ms.get_z_for_particle_in_loop(x_k_for_run)
    z_k_debug = atrapp.get_z_for_states_in_times_through_torch(x_k_for_run)
    assert np.all( np.abs(z_k_debug - z_k) < 1e-10)
    if 0:
        fig, axs = plt.subplots(1, 3)
        # plt.sca(axes[1, 1])
        axs[0].imshow(z_k[0])
        axs[1].imshow(z_k_debug[0])
        plt.show(block=False)
    if not ms.cheat_dont_add_noise_to_meas:
        z_k += np.random.normal(0,ms.v_var,z_k.shape)
    #fig, axs = plt.subplots()
    ## plt.sca(axes[1, 1])
    #plt.xticks(range(ms.nof_s_x), ms.center[0] - ms.nensor_size[0] / 2 + ms.dt * np.arange(ms.nof_s_x), color='k')
    #plt.yticks(range(ms.nof_s_y), ms.center[1] - ms.nensor_size[1] / 2 + ms.dt * np.arange(ms.nof_s_y), color='k')
    #axs.imshow(z_k[0])
    #plt.show(block=True)

    return np.reshape(x_k[:nof_steps, :nof_targs],(nof_steps,nof_targs, 4)), np.reshape(z_k[:nof_steps],(nof_steps, 13,13))


if __name__ == '__main__':


    model = AtrappModel(opt=ms, sensor_params=ms.sensor_params, mm_params=ms.mm_params, device=ms.device_str)
    pfbh = PfBatchHandler(model=model, opt=ms)
    train_nn1 = 0
    train_nn2 = 0
    train_nn3 = 0

    steps = ms.nof_steps
    forward_num_particles = ms.nof_parts
    backwards_num_particles = ms.nof_parts
    nof_iteraions = 10
    P0 = ms.Q*np.max(ms.sensor_size)
    mu0 = [ms.center[0], 0, ms.center[1], 0]
    Q = ms.Q
    R = ms.v_var
    nof_targs = ms.nof_targets
    print("nof_steps_for_sim: "+str(ms.nof_steps_for_sim))
    print("nof_targs: "+str(nof_targs))
    print("num_particles: "+str(forward_num_particles))

    full_file_path = "../pf/particles/pt_parts" + str(9000) + "_tars" + str(1) + "_steps" + str(100) + ".npy"
    full_file_path = "../pf/particles/pt_parts" + str(1000) + "_tars" + str(1) + "_steps" + str(100) + ".npy"

    total_runs_ospa = 0
    total_runs_ospa_from10 = 0
    total_runs_ospa_from20 = 0
    total_time_start = time.time()
    model.eval()
    val_loss, val_dice, n_total, _n_batch = 0, 0, 0, nof_iteraions
    val_loss_ts, val_dice_ts = 0, 0
    nn3_time_per_batch_single_step, atrapp_time_per_batch_single_step, meas_time_per_batch_single_step = 0, 0, 0

    for iter in np.arange(nof_iteraions):
        (x, y) = generate_dataset(full_file_path, nof_targs, ms.nof_steps_for_sim)
        x = torch.unsqueeze(torch.Tensor(x).to(ms.device), 0)
        y = torch.unsqueeze(torch.Tensor(y).to(ms.device), 0)
        curr_batch_size, curr_nof_steps = x.shape[0], x.shape[1]

        model.reset_before_batch([False, False, False], x)
        loss_b_ts, ospa_batch_b_ts, (atrapp_time, nn3_time, meas_time) = pfbh.get_batch_loss(z=y, x=x)
        loss = torch.sum(loss_b_ts, dim=1) / curr_nof_steps
        ospa_batch = torch.sum(ospa_batch_b_ts, dim=1) / curr_nof_steps
        loss = torch.sum(loss) / curr_batch_size
        # val_dice = np.sum(-ospa_batch.cpu().detach().numpy()*(n_total +curr_batch_size))/curr_batch_size
        dice_item = np.sum(-ospa_batch.cpu().detach().numpy()) / curr_batch_size
        loss_item = loss.item()
        val_loss += loss_item * curr_batch_size
        val_dice += dice_item * curr_batch_size
        val_loss_ts += np.sum(loss_b_ts.cpu().detach().numpy(), axis=0)  # * curr_batch_size
        val_dice_ts += np.sum(ospa_batch_b_ts.cpu().detach().numpy(), axis=0)  # * curr_batch_size
        n_total += curr_batch_size
        nn3_time_per_batch_single_step += nn3_time / curr_nof_steps
        atrapp_time_per_batch_single_step += atrapp_time / curr_nof_steps
        meas_time_per_batch_single_step += meas_time / curr_nof_steps

    res = val_loss / n_total, val_dice / n_total, val_loss_ts / n_total, val_dice_ts / n_total, iter + 1, (atrapp_time_per_batch_single_step / (iter + 1), nn3_time_per_batch_single_step / (iter + 1), meas_time_per_batch_single_step / (iter + 1))
    print(res)

    fig, axs = plt.subplots()
    # plt.sca(axes[1, 1])
    axs.imshow(np.zeros((100,100)))
    plt.show(block=True)
