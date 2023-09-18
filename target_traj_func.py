import numpy as np
import torch
import os

def make_x0(opt, nof_steps, speed_factor=1, fixed_initial_vels = True):
    x0 = np.zeros((opt.state_vector_dim))
    for xy in [0, 1]:
        x0[2*xy] = np.random.uniform(low=opt.center[xy] - opt.sensor_size[xy]/2, high =opt.center[xy] + opt.sensor_size[xy]/2,size=1)
        if fixed_initial_vels:
            x0[2 * xy + 1] = np.random.choice([0.1, -0.1], p=[0.5, 0.5])
        else:
            direction = 1 if x0[2*xy] < opt.center[xy] else -1
            max_speed =  direction*(opt.sensor_size[xy]/2+np.abs(x0[0]-opt.center[xy]))/(opt.tau*nof_steps)
            min_speed = -direction*(opt.sensor_size[xy]/2-np.abs(x0[0]-opt.center[xy]))/(opt.tau*nof_steps)
            x0[2*xy+1] = np.random.uniform(low=min_speed*speed_factor,high=max_speed*speed_factor)
    return x0

def get_target_traj(nof_steps, x0, F, Q, max_iters=1000, opt=None, mm=None):
    target_traj = np.zeros((nof_steps+1,opt.state_vector_dim))
    target_traj[0] = x0
    suceeded = False
    trial_idx = 0
    while not suceeded:
        trial_idx += 1
        if trial_idx > max_iters:
            return False
        for step_idx in np.arange(nof_steps):
            weighted_avg_loc = torch.reshape(torch.tensor(target_traj[step_idx,(0,2)], device = 'cpu'),(1,1,1,2))
            weighted_avg_vel = torch.reshape(torch.tensor(target_traj[step_idx,(1,3)], device = 'cpu'),(1,1,1,2))
            new_weighted_avg_loc, new_weighted_avg_vel = mm.advance_locations(False, weighted_avg_loc.to(opt.device_str), weighted_avg_vel.to(opt.device_str), opt.device_str, print_seed=False, print_grad=True)
            target_traj[step_idx+1,(0,2)] = new_weighted_avg_loc.cpu().detach().numpy()
            target_traj[step_idx+1,(1,3)] = new_weighted_avg_vel.cpu().detach().numpy()
            bad_step = False
            for xy in [0, 1]:
                if target_traj[step_idx, 2 * xy] < opt.center[xy] - opt.sensor_size[xy] / 2 or target_traj[step_idx, 2 * xy] > opt.center[xy] + opt.sensor_size[xy] / 2:
                    bad_step = True
                    break
            if bad_step: break
        if bad_step:
            continue
        suceeded = True
    return target_traj

#makes nof_targets trjectories using get_target_traj and make_x0
def make_particle_traj(nof_targets, nof_steps, fixed_initial_vels, opt=None, mm=None, prnt_str=""):
    x_t = np.zeros((nof_steps + 1, nof_targets, opt.state_vector_dim))
    for target_idx in np.arange(nof_targets):
        x0 = make_x0(opt, nof_steps, fixed_initial_vels=fixed_initial_vels)
        target_traj = False
        while 1:
            target_traj = get_target_traj(nof_steps, x0, opt.mm_params.F, opt.mm_params.Q, opt=opt, mm=mm)
            if type(target_traj) != bool:
                break
            else:
                print("dfsdfgs")
                x0 = make_x0(opt, nof_steps, fixed_initial_vels=fixed_initial_vels)
        x_t[:,target_idx] = target_traj
    return x_t

def make_parts_trajs(nof_parts, nof_targets, nof_steps, fixed_initial_vels, opt=None, mm=None, prnt_str=""):
    xs = np.zeros((nof_parts, nof_steps + 1, nof_targets, opt.state_vector_dim))
    for part_idx in np.arange(nof_parts):
        xs[part_idx] = make_particle_traj(nof_targets, nof_steps, fixed_initial_vels, opt=opt, mm=mm, prnt_str=prnt_str)
        if np.mod(part_idx,100)==0:
            print("on: " + prnt_str + "made full particle part_idx: " + str(part_idx)+" of: "+ str(nof_parts))
    return xs

def make_trajs_mm(opt, mm):
    do_run_get_particles = True
    #nof_parts = 1000
    nof_targets = 1
    nof_steps = 100
    fixed_initial_vels = True
    nof_sets = 10

    set_strs = "test2", "test3"
    parts_per_set = 10000, 10000
    parts_per_set = 3, 3

    if do_run_get_particles:
        for set_str, nof_parts in zip(set_strs, parts_per_set):
            for set_idx in np.arange(nof_sets):
                folder_path = "./particles/orig_motion/" +set_str+'_sets/'
                file_path_str = folder_path+ set_str+'_set'+str(set_idx)+'_'+str(nof_parts) +"parts_" + str(nof_targets) + "targs_" + str(nof_steps) +"steps" + ".npy"
                try:
                    with open(file_path_str, 'rb') as f:
                        x_ts = np.load(f)
                        f.close()
                except:
                    #exit()
                    x_ts = make_parts_trajs(nof_parts, nof_targets, nof_steps, fixed_initial_vels, opt=opt, mm=mm, prnt_str=file_path_str)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    with open(file_path_str, 'wb') as f:
                        np.save(f, x_ts)