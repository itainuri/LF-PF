import argparse
from Instructor import *
import os
import torch
import numpy as np
import copy
import matplotlib
from BatchMaker import PfDataVars, PfBatchMaker
from OptConfig import OptConfig
from MotionModel import MotionModelParams
from SensorModel import SensorParams, SensorModel

from sys import platform
if not (platform == "linux" or platform == "linux2") and not os.path.exists("/content"):
    matplotlib.use("Qt5Agg")

torch.set_default_tensor_type('torch.FloatTensor')
torch.set_default_dtype(torch.float64)

config=OptConfig()
def make_instructor_args(config):
    ## if running on google colab using cmdline %run external/train2.py needs to add -f, if !python train2.py dont need
    ##if g_colab: sys.argv = ['-f']
    parser = argparse.ArgumentParser()
    ''' For dataset '''
    parser.add_argument('--path2proj', default=config.path2proj, type=str, help='')
    parser.add_argument('--state_vector_dim', default=config.state_vector_dim, type=int)
    parser.add_argument('--nof_steps', default=config.nof_steps, type=int)
    parser.add_argument('--batch_size', default=config.batch_size, type=int)
    parser.add_argument('--nof_batches_per_epoch', default=config.nof_batches_per_epoch, type=int)
    parser.add_argument('--nof_parts', default=config.nof_parts, type=int)
    parser.add_argument('--nof_epochs', default=config.nof_epochs, type=int)
    parser.add_argument('--seed', default=config.seed, type=int, help='seed, seed=0 -> random')
    parser.add_argument('--is_random_seed', default=int(config.is_random_seed), type=int, help='0, 1')
    parser.add_argument('--device_str', default=config.curr_device_str, type=str, help='cpu, cuda')
    parser.add_argument('--make_batch_device_str', default=config.make_batch_device_str, type=str, help='cpu, cuda')
    parser.add_argument('--model_mode', default=config.model_mode, type=str, help='shoudl be attention')
    ''' For APP model '''
    parser.add_argument('--lh_sig_sqd', default=config.lh_sig_sqd, type=float)
    parser.add_argument('--ospa_p', default=config.ospa_p, type=int)
    parser.add_argument('--ospa_c', default=config.ospa_c, type=int)
    parser.add_argument('--skip_nn3', default=config.skip_nn3, type=int)
    ''' Cheats '''
    parser.add_argument('--cheat_first_particles', default=config.cheat_first_particles, type=int)
    parser.add_argument('--cheat_first_locs_only_half_cheat', default=config.cheat_first_locs_only_half_cheat, type=int)
    parser.add_argument('--locs_half_cheat_var', default=config.locs_half_cheat_var, type=float)
    parser.add_argument('--cheat_first_vels', default=config.cheat_first_vels, type=int)
    parser.add_argument('--cheat_dont_add_noise_to_meas', default=config.cheat_dont_add_noise_to_meas, type=int)
    ''' For motopn and sensor models '''
    parser.add_argument('--snr0', default=config.snr0, type=float, help='for the sensor model')
    parser.add_argument('--snr_half_range', default=config.snr_half_range, type=float, help='for the sensor model')
    parser.add_argument('--d0', default=config.d0, type=float, help='for the sensor model')
    parser.add_argument('--center', default=config.center, type=float, help='for the sensor model')
    parser.add_argument('--sensor_size', default=config.sensor_size, type=float, help='for the sensor model')
    parser.add_argument('--v_var', default=config.v_var, type=float, help='for the sensor model')
    parser.add_argument('--dt', default=config.dt, type=float, help='for the sensor model')
    parser.add_argument('--eps', default=config.eps, type=float, help='for the sensor model')
    parser.add_argument('--tau', default=config.tau, type=float, help='for the sensor model')
    parser.add_argument('--sig_u', default=config.sig_u, type=float, help='for the sensor model')
    parser.add_argument('--sensor_active_dist', default=config.sensor_active_dist, type=float, help='sensors further than this distance are neglected')
    parser.add_argument('--do_inaccurate_sensors_locs', default=int(config.do_inaccurate_sensors_locs), type=int, help='True/False')
    parser.add_argument('--inaccurate_sensors_locs_offset_var', default=config.inaccurate_sensors_locs_offset_var, type=float, help='')
    parser.add_argument('--lost_targ_dist', default=config.lost_targ_dist, type=float, help='stopping batch if 1 target is lost')
    ''' For environment '''
    parser.add_argument('--proj2datasets_path', default=config.proj2datasets_path, type=str, help='')
    parser.add_argument('--proj2ckpnts_load_path', default=config.proj2ckpnts_load_path, type=str, help='')
    parser.add_argument('--att_load_checkpoint', default=int(config.att_load_checkpoint), type=int)
    parser.add_argument('--attention_checkpoint', default=config.att_state_dict_to_load_str, type=str)
    parser.add_argument('--dont_print_progress', default=int(config.dont_print_progress), type=int, help='True/False')
    parser.add_argument('--make_new_trajs', default=int(config.make_new_trajs), type=int, help='True/False')
    ''' For inference '''
    parser.add_argument('--inference_do_compare', default=int(config.inference_do_compare), type=int)
    parser.add_argument('--inference_mode', default=config.inference_mode, type=str, help='paint, eval')

    opt = parser.parse_args()

    opt.model_mode = modelmode[opt.model_mode]
    opt.inference_mode = inferencemode[opt.inference_mode]
    if opt.is_random_seed:
        opt.seed = np.random.random_integers(np.power(2, 30))

    torch.manual_seed(seed=opt.seed)
    torch.manual_seed(opt.seed)
    np.random.seed(seed=opt.seed)
    np.random.seed(opt.seed)
    attention_network = modelmode[opt.model_mode] in {"attention"}
    if modelmode[opt.model_mode] == "attention":
        opt.model_name = 'att'
    else:
        assert 0, "not supported"
    test_batch_maker = None
    assert opt.snr0 >= opt.snr_half_range
    sensor_params = SensorParams(snr0=opt.snr0, snr_half_range=opt.snr_half_range,
                                 d0=opt.d0, center=opt.center, sensor_size=opt.sensor_size, v_var=opt.v_var,
                                 dt=opt.dt, eps=opt.eps, sensor_active_dist=opt.sensor_active_dist, lh_sig_sqd=opt.lh_sig_sqd)
    bm_sm_sensor_params = copy.deepcopy(sensor_params)

    sensor_params.set_z_coo_xy_for_all_sensors_without_noise(device = opt.device_str)
    sensors_locs_dir_str = './sensors_locs/'
    sensor_locs_noise_str = "sensors locations noises"
    create_new = 0# change to 1 to make new sensor array offsets, o/w loads saved offsets
    if opt.do_inaccurate_sensors_locs:
        if create_new:
            all_z_coo_xy, assumed_all_z_coo_xy  = bm_sm_sensor_params.make_return_z_coo_xy_for_all_sensors(
                add_noise_to_sensors_locs = opt.do_inaccurate_sensors_locs, offset_var = opt.inaccurate_sensors_locs_offset_var, device = opt.make_batch_device_str)
            sensors_locs_sav_path = sensors_locs_dir_str + modelmode[opt.model_mode] + '/' + 'locs_var={:.2f}.pt'.format(opt.inaccurate_sensors_locs_offset_var)
            torch.save((all_z_coo_xy, assumed_all_z_coo_xy), sensors_locs_sav_path)
            sensor_locs_noise_str += " saved, string: " + sensors_locs_sav_path
        else:
            sensors_locs_load_path =  sensors_locs_dir_str + modelmode[opt.model_mode] + '/' + 'locs_var={:.2f}.pt'.format(opt.inaccurate_sensors_locs_offset_var)
            all_z_coo_xy, assumed_all_z_coo_xy = torch.load(sensors_locs_load_path, map_location='cpu')
            sensor_locs_noise_str += " loaded, string: " + sensors_locs_load_path
    else:
        all_z_coo_xy, assumed_all_z_coo_xy = bm_sm_sensor_params.make_return_z_coo_xy_for_all_sensors(
            add_noise_to_sensors_locs=False, offset_var=opt.inaccurate_sensors_locs_offset_var, device=opt.make_batch_device_str)
        sensors_locs_sav_path = sensors_locs_dir_str + modelmode[opt.model_mode] + '/' + 'locs_var={:.2f}.pt'.format(1)
    print(sensor_locs_noise_str+", max offset: " + str(torch.max(all_z_coo_xy - assumed_all_z_coo_xy)) + ", min offset: " + str(torch.min(all_z_coo_xy - assumed_all_z_coo_xy)))
    all_z_coo_xy, assumed_all_z_coo_xy = all_z_coo_xy.to(opt.device_str), assumed_all_z_coo_xy.to(opt.device_str)
    bm_sm_sensor_params.set_z_coo_xy(all_z_coo_xy, assumed_all_z_coo_xy)
    true_sensor_model = SensorModel(sensor_params=bm_sm_sensor_params)
    true_sensor_model.reset(bm_sm_sensor_params)
    opt.true_sensor_model = true_sensor_model
    mm_params = MotionModelParams(tau = opt.tau, sig_u=opt.sig_u)
    opt.sensor_params = sensor_params
    opt.mm_params = mm_params

    if attention_network:
        nof_ts_ds = 100
        nof_parts_test_ds = 1000
        test_data_paths_list = []
        for set_idx in np.arange(10):
            test_data_paths_list.append("test_set" + str(set_idx)+"_" + str(nof_parts_test_ds) + "parts_" + str(1) + "targs_" + str(nof_ts_ds) + "steps.npy")
        epoch_sizes = [opt.nof_steps, opt.batch_size, opt.nof_batches_per_epoch]
        att_path_2imgs_dir = opt.path2proj + opt.proj2datasets_path
        att_test_data_vars  = PfDataVars(path2data=att_path_2imgs_dir+"/test_sets2/", data_paths_list=test_data_paths_list, epoch_sizes=epoch_sizes)

    if modelmode[opt.model_mode] in {"attention"}:
        test_batch_maker = PfBatchMaker(opt=opt, data_vars=att_test_data_vars)
    return opt, test_batch_maker

def _mp_fn(index, args):
    opt, test_batch_maker = args
    args = opt, test_batch_maker
    ins = Instructor(*args)
    ins.start()

if __name__ == '__main__':
    print(" __name__ == __main__, running instructor")
    args = make_instructor_args(config)
    time_str = time.strftime("%Y_%m_%d_%H%M%S", time.localtime())
    print("curr local time: "+time_str)
    _mp_fn(0, args)

    fig, axs = plt.subplots()
    axs.imshow(np.zeros((100, 100)))
    plt.show(block=True)
    fff = 9