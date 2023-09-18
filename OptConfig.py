
class FrozenClass(object):
    __isfrozen = False
    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError( "%r is a frozen class" % self )
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True

class OptConfig(FrozenClass):
    __isfrozen = False
    def __init__(self):

        self.model_mode = "attention" # to add different settings
        self.curr_device_str = 'cuda'  # device
        # self.curr_device_str = 'cpu'
        self.make_batch_device_str = 'cpu'
        # self.make_batch_device_str = 'cuda'
        self.make_new_trajs = 0 # no simulation - creates new trajectories
        self.state_vector_dim = 4
        self.nof_steps =100 # number of time steps for simulation
        self.nof_parts = 100 # number of particles for simulation
        self.do_inaccurate_sensors_locs = 1 # 0-calibrated sensors setting 1-miscalibrated setting
        self.inaccurate_sensors_locs_offset_var = 1.0 # sensors locations offsets variance (ofssets in x and y are nomally distrbuted), loads offsets from "/sensors_locs", if wants to make new offsets change to create_new=1 in "ins_runner.py"
        self.skip_nn3 = 0 # APP if skip NA-APF if not
        self.dont_print_progress = 0 # prints batch/(total batches) wint ">>"
        self.is_random_seed = 0
        self.seed = 6
        self.inference_do_compare = 1 # if skip_nn3=0 runs a second run with skip_nn3=1 and compares results
        self.inference_mode = 'paint' # paints trajectories and sensors
        self.inference_mode = 'eval' # does inference without painting
        self.cheat_first_particles = 1 # initial particles are accorsing to true state
        self.cheat_first_locs_only_half_cheat = 1 # adds small variance to inital particles according to locs_half_cheat_var
        self.locs_half_cheat_var = 0.01 # adds small variance to particles (for cheat_first_locs_only_half_cheat)
        self.cheat_first_vels = 1 #initial particles velocities are according to true state
        self.att_load_checkpoint = 1 #loading pretrained weights
        self.batch_size = 9
        self.nof_batches_per_epoch = 10000 / self.batch_size # if 0 uses all dataset
        self.nof_epochs = 1
        self.lost_targ_dist = 10.0 # for internal use in training
        if self.inference_mode == 'eval':
            self.batch_size = 10
            self.nof_batches_per_epoch = 2
            self.nof_epochs = 1
            # on original paper APF(APP) was tested with:
            self.cheat_first_particles = 1
            self.cheat_first_locs_only_half_cheat = 0
            self.cheat_first_vels = 1
        elif self.inference_mode == 'paint':
            self.batch_size = 4
            self.nof_batches_per_epoch = 3
            self.nof_epochs = 1
        self.sig_u = 0.1 # for motion model
        self.tau = 1 # time interval bwtween timesteps
        self.snr0 = 20. # for sensor model SNR=snr0/v_var
        self.snr_half_range = 0. # to have different SNRs on sensor model on same run
        self.d0 = 5.  # for sensor model
        self.center = [100, 100] #center of sensor in meters
        sensor_width = 120
        self.sensor_size = [sensor_width, sensor_width] #dims of sensor in meters
        self.sensor_active_dist = 20 # valid pixels maximum distance from average particle
        self.v_var = 1 # noise variance of the sensor model
        self.dt = 10 # for sensor model
        self.eps = 1e-18 # for sensor model
        self.ospa_p = 2. #power of OSPA on loss
        self.ospa_c = 100000000000.  # cutoff for loss OSPA (vor dice cutoff=10.0)
        self.lh_sig_sqd = 1 #variance of gaussian when comparing 2 pixels values in measurement
        self.cheat_dont_add_noise_to_meas = 0 # sensor doesn't have noise
        self.path2proj = ""
        self.proj2datasets_path = "../ltbd0/particles/orig_motion"
        self.proj2ckpnts_load_path = './state_dict/'
        if self.do_inaccurate_sensors_locs:
            self.att_state_dict_to_load_str = "mismatched_sensor_array_NN_weights.pt"
        else:
            self.att_state_dict_to_load_str = "accurate_sensor_array_NN_weights.pt"

        self._freeze() # no new attributes after this point.


