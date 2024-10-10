# LF-APF official
inference implementation of the Neural Augmented Auxiliary Particle Filter, from the paper: 

Itai Nuri and Nir Shlezinger, 2023, "NEURAL AUGMENTED AUXILIARY PARTICLE FILTERS WITH APPLICATION TO RADAR TRACK-BEFORE-DETECT"

In the paper we propose a DNN-aided Auxiliary Particle Filter algorithm designed to facilitate operation 
with a reduced number of particles and with possible mismatches and approximation errors
in the observation model. The DNN architecture applies a fix to all particles and weights and so copes with changing number of particles 
and can be integrated to most Particle Filter algorithms and in different (and multiple) stages.
the paper shows a 2-3 times reduction in number of particles, and an improvement unachievable by an increase in number of particles in a mismatched observation model case.  

# Table of Contents
- [Introduction](#introduction)
  * [Terminology](#Terminology)
- [python_code directory](#python_code-directory)
  * [Data directories](#Data-directories)
    + [particles](#particles)
    + [sensors_locs](#sensors_locs)
    + [state_dict](#state_dict)
  * [Python Files](#Python-Files)
- [Simulation](#Simulation)
  * [Environment Main Packages Versions](#Environment-Main-Packages-Versions)
  * [Paths definitions](#Paths-Definitions)
  * [Execution](#Execution)
  * [Simulation Flags](#Simulation-Flags)
  
# Introduction
the provided code supports inferencing the experiments as described on the paper with the saved weights and sensors offsets. 
It presents tracking results (and compares) APF and NA-APF and illustrates tracking accuracy examples and sensors response.
it also contains the test set trajectories and enables the creations of new trajectories according to the motion model.

# Terminology
- APF - Auxiliary Particle Filter 
- APP - Auxiliary Parallel Partition (PF) (single target): APF + Kalman Filter for velocities 
- NA-APF - Neural Augmented APF: APF + Kalman Filter + particles and weights correction DNN


# python_code directory
Contains all files needed to run simulations. The structure of the data and python classes are designed for easy user specific adaptations. 
To adjust the code edit the content of the functions of the different classes described next.  

## Data directories 
includes the ground truth trajectories, the sensors' locations offsets, and the DNN weights.

### particles 
Includes the ground truth targets trajectories, the 10,000 testing trajectories of the paper. 
more trajectories can be created and saved using this code, 
new trajectories names and paths configurations are set in "target_traj_func.py" 
### sensors_locs 
Includes the mismatched configuration sensors offsets, in which the respective DNN weights were trained on. 
new offsets can be created with the respective configuration flags and by setting create_new flag in "ins_runner.py". 

### state_dict
Includes the saved weights for both accurate and mismatched sensors settings. 
should be loaded for optimal accuracy on respective settings.
## Python Files
* OptConfig.py - default simulation configurations.
* ins_runner.py - parses command line and OptConfig.py for the simulation flags and starts simulation on Instructor.
* MotionModel - used to propagte particles from one step (apf) and to cretae new trajectories (target_traj_func).
* SensorModel - creates a sensor response to a target with specific state. used for measuring (apf) and for creating sensor input on simulation (BatchMaker).
* target_traj_func.py - creates new ground truth trajectories.
* BatchMaker.py - loads the ground truth trajectories for the simulation from the files, and creates input and expected output pairs.
* Instructor.py - contains the simulation class Instructor, runs PfBatchHandler.
* KalmanFIlter.py - runs a single step of Kalman Filter.
* apf.py - runs a single step of the APF or NA-APF.
* AppModel.py - runs a single iteration of APF/NA-APF + Kalman Fiter.
* NN_blocks.py - holds the DNN modules.
* BatchData.py - holds the particles and weights of an iteration (used in PfBatchHandler).
* PfBatchHandler.py - runs a full batch trajectory and calculates OSPA and loss.

# Simulation
copy the content of the directory "python_code" to a directory in your project directory.
## Environment Main Packages Versions 

    python	3.8.13
    pytorch	1.11.0
    numpy	1.23.4
    matplotlib	3.6.1
## Paths Definitions
default paths are in the configurations file, OptConfig.py:

model_mode, path2proj, proj2datasets_path, proj2ckpnts_load_path, att_state_dict_to_load_str

## Execution
go to the main directory containing "ins_runner.py" and run (for example):

python ins_runner.py --make_new_trajs 0 --batch_size 10 --nof_batches_per_epoch 2  --nof_epochs 1 --nof_steps 100 --nof_parts 100 --snr0 20.0 --snr_half_range 0 --sensor_active_dist 20 --do_inaccurate_sensors_locs 0 --inaccurate_sensors_locs_offset_var 1.0 --cheat_first_particles 1 --cheat_first_locs_only_half_cheat 1 --locs_half_cheat_var 0.01 --cheat_first_vels 1 --path2proj "./" --proj2datasets_path "./particles/orig_motion" --proj2ckpnts_load_path "./state_dict/" --is_random_seed 0 --seed 18 --device_str 'cuda' --make_batch_device_str 'cpu' --inference_mode 'paint' --dont_print_progress 1 --skip_nn3 0 --model_mode 'attention' --att_load_checkpoint 1 --attention_checkpoint 'accurate_sensor_array_NN_weights.pt'

## Simulation Flags

* model_mode  # only supports "attention", to add different settings add additional modes
* curr_device_str # device 'cuda' or 'cpu'
* make_batch_device_str # make batch device string 'cuda' or 'cpu'
* make_new_trajs # 1- creates new trajectories, 0-runs simulation
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
* inference_mode # 'paint'-paints trajectories and sensors, 'eval'-inferences without painting
* cheat_first_particles # 1-initial particles are according to true state, 0-unifiormely distruted in the sensor field of view
* cheat_first_locs_only_half_cheat # adds small variance to initial particles according to locs_half_cheat_var
* locs_half_cheat_var # adds small variance to particles (for cheat_first_locs_only_half_cheat)
* cheat_first_vels # initial particles velocities are according to true state
* att_load_checkpoint # 1-loading pretrained DNN weights, 0- starting with random DNN weights
* batch_size # batch size
* nof_batches_per_epoch # if 0 uses all dataset
* nof_epochs # number of epochs
* lost_targ_dist = 10.0 # for internal use on training
* sig_u # for motion model
* tau # time interval between timesteps
* snr0 = 20. # for sensor model SNR=snr0/v_var
* snr_half_range # to have changing SNRs on sensor model on same run, SNR uniformely distributed in snr0+-snr_half_range
* d0 # for sensor model as described in paper
* center # center positions of sensor x and y in meters
* sensor_size = [sensor_height sensor_width] # dims of sensor in meters
* sensor_active_dist # valid pixels maximum distance from average particle, further sensors are ignored on paricle weighting.
* v_var = 1 # noise variance of the sensor model
* dt = 10 # for sensor model
* eps = 1e-18 # for sensor model
* ospa_p # power of OSPA on loss
* ospa_c # cutoff for loss OSPA (for dice cutoff=10.0)
* lh_sig_sqd # variance of gaussian when comparing 2 pixels values in measurement
* cheat_dont_add_noise_to_meas # sensor response doesn't have noise
* path2proj, proj2datasets_path, proj2ckpnts_load_path # paths for all loaded data
* att_state_dict_to_load_str # DNN saved weights file name ("mismatched_sensor_array_NN_weights.pt" or "accurate_sensor_array_NN_weights.pt")

