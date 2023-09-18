
NEURAL AUGMENTED AUXILIARY PARTICLE FILTERS WITH APPLICATION TO RADAR TRACK-BEFORE-DETECT"
--
this repository includes inference implementation of the paper: "NEURAL AUGMENTED AUXILIARY PARTICLE FILTERS WITH APPLICATION TO RADAR TRACK-BEFORE-DETECT" (Itai Nuri and Nir Shlezinger)

- currently the repository only supports inferencing APP and NA-APF.
- the saved weights for both accurate and mismatched sensors settings are included and loaded on their respective setting according to the configurations.
- the mismatched configuration sensors offsets, in which the respective DNN weights were trained on, are also included and are loaded on that configuration.
- the 10,000 testing trajectories of the paper are included, more trajectories can be created using this code.

Terminology:
--
- APF - Auxiliary Particle Filter 
- APP - Auxiliary Parallel Partition (PF) (single target): APF + Kalman Filter for velocities 
- NA-APF - Neural Augmented APF: APF + Kalman Filter + particles and weights correction DNN

Main Packages Versions
--
    python	3.8.13
    pytorch	1.11.0
    numpy	1.23.4
    matplotlib	3.6.1

Simulation Configuration Flags
--
* model_mode  # only supports "attention", to add different settings add additional modes
* curr_device_str # device 'cuda' or 'cpu'
* make_batch_device_str # make batch device string 'cuda' or 'cpu'
* make_new_trajs # 1-no simulation and creates new trajectories, 0-runs simulation
* state_vector_dim # dimnsion of the state vector
* nof_steps  # number of time steps for simulation
* nof_parts # number of particles for simulation
* do_inaccurate_sensors_locs # 0-calibrated sensors setting 1-miscalibrated setting
* inaccurate_sensors_locs_offset_var # sensors locations offsets variance (offsets in x and y are normally distributed), loads offsets from "/sensors_locs", if wants to make new offsets change to create new=1 in "ins_runner.py"
* skip_nn3 # 1-APP, 0-NA-APF
* dont_print_progress # prints batch/(total batches) ratio with ">>"
* is_random_seed # 1-random seed, 0-seed="seed" (same seed for python, pytorch and numpy)
* seed # seed to use for python, pytorch and numpy (if  is_random_seed=0)
* inference_do_compare # if skip_nn3=0 runs a second run with skip_nn3=1 and compares results
* inference_mode # 'paint'-paints trajectories and sensors, 'eval'-does inference without painting
* cheat_first_particles # 1-initial particles are according to true state, 0-unifiormely distruted in the sensor field of view
* cheat_first_locs_only_half_cheat # adds small variance to initial particles according to locs_half_cheat_var
* locs_half_cheat_var # adds small variance to particles (for cheat_first_locs_only_half_cheat)
* cheat_first_vels # initial particles velocities are according to true state
* att_load_checkpoint # 1-loading pretrained weights, 0- starting with random weights
* batch_size # batch size
* nof_batches_per_epoch # if 0 uses all dataset
* nof_epochs # number of epochs
* lost_targ_dist = 10.0 # for internal use in training
* sig_u # for motion model
* tau # time interval between timesteps
* snr0 = 20. # for sensor model SNR=snr0/v_var
* snr_half_range # to have different SNRs on sensor model on same run.  uniformely distributed in snr0+-snr_half_range
* d0 # for sensor model as described in paper
* center # centers of sensor x and y in meters
* sensor_size = [sensor_height sensor_width] # dims of sensor in meters
* sensor_active_dist # valid pixels maximum distance from average particle
* v_var = 1 # noise variance of the sensor model
* dt = 10 # for sensor model
* eps = 1e-18 # for sensor model
* ospa_p # power of OSPA on loss
* ospa_c # cutoff for loss OSPA (for dice cutoff=10.0)
* lh_sig_sqd # variance of gaussian when comparing 2 pixels values in measurement
* cheat_dont_add_noise_to_meas # sensor doesn't have noise
* path2proj, proj2datasets_path, proj2ckpnts_load_path # paths for all loaded data
* att_state_dict_to_load_str # "mismatched_sensor_array_NN_weights.pt" or "accurate_sensor_array_NN_weights.pt"


default configurations are in OptConfig.py,
running command example:

python ins_runner.py --make_new_trajs 0 --batch_size 10 --nof_batches_per_epoch 2  --nof_epochs 1 --nof_steps 100 --nof_parts 100 --snr0 20.0 --snr_half_range 0 --sensor_active_dist 20 --do_inaccurate_sensors_locs 0 --inaccurate_sensors_locs_offset_var 1.0 --cheat_first_particles 1 --cheat_first_locs_only_half_cheat 1 --locs_half_cheat_var 0.01 --cheat_first_vels 1 --path2proj "./" --proj2datasets_path "./particles/orig_motion" --proj2ckpnts_load_path "./state_dict/" --is_random_seed 0 --seed 18 --device_str 'cuda' --make_batch_device_str 'cpu' --inference_mode 'paint' --dont_print_progress 1 --skip_nn3 0 --model_mode 'attention' --att_load_checkpoint 1 --attention_checkpoint 'accurate_sensor_array_NN_weights.pt'
