""" Particle filtering for a trivial model
    Also illustrates that the """

import numpy
import matplotlib.pyplot as plt
#import simulator as simulator
import numpy as np
#import models as ms
import Atrapp as atrapp
from KalmanFilter import KalmanFilter
from MotionModel import MotionModel
from SensorModel import SensorModel

import time
import copy
import torch
import torch.nn as nn

ms_do_debug_prints = False

##########################################################################
class AtrappModel(nn.Module):
    """ x_{k+1} = x_k + v_k, v_k ~ N(0,Q)
        y_k = x_k + e_k, e_k ~ N(0,R),
        x(0) ~ N(0,P0) """

    def __init__(self, opt, sensor_params, mm_params, device):

        super(AtrappModel, self).__init__()
        self.device = device
        self.F = torch.from_numpy(opt.mm_params.F).to(self.device)
        self.kf = KalmanFilter(device = self.device)
        self.mm = MotionModel(device = self.device, opt=opt)
        self.sm = SensorModel(device = self.device, opt=opt)

        #self.nn1 = []
        #self.nn1.append(NN1(nof_parts=opt.nof_parts, nof_targs=opt.nof_targs, skip=opt.skip_nn1, is_nn3=False).to(self.device))
        #self.nn1.append(NN1(nof_parts=opt.nof_parts, nof_targs=opt.nof_targs, skip=opt.skip_nn1, is_nn3=False).to(self.device))

        #self.nn3 = NN3(nof_parts=opt.nof_parts, nof_targs=opt.nof_targs, skip=opt.skip_nn3, is_nn3=True).to(self.device)
        self.is_TRAPP = opt.is_TRAPP
        self.opt = opt
        self.sensor_params = sensor_params
        self.mm_params = mm_params
        # Q - process noise
        # R - observation noise
        #self.kf.set_dynamics(A=ms.F, C=np.eye(ms.F.shape[0]), Q=ms.Q, R=np.matmul(np.matmul(ms.Q,ms.F),np.transpose(ms.Q,(1,0)))) #2
        #self.kf.set_dynamics(A=ms.F, C=np.eye(ms.F.shape[0]), Q=ms.Q, R=(ms.nof_parts)*ms.Q) #2
        #self.kf.set_dynamics(A=np.eye(2), C=np.eye(2), Q=0.01*np.eye(2), R=0.1*np.eye(2)) #3
        #self.kf.set_dynamics(A=np.eye(2), C=np.eye(2), Q=1000*np.eye(2), R=100*np.eye(2)) #4
        #self.kf.set_dynamics(A=np.eye(2), C=np.eye(2), Q=ms.Q[0,1]*np.eye(2), R=2*0.25*ms.tau*np.eye(2)) #5

        if 0:
            C0 = np.zeros((2,4))
            C0[0, 0] = 1
            C0[1, 2] = 1
            A0 = self.F # TODO inputting the new advances time so dousnt need to update, in general needs to update
            Q0 = 1e-5*np.eye(4)#ms.Q
            Q0[0,0] = 1e-6
            Q0[2,2] = 1e-6
            #Q0 = ms.Q
            R0 = 1 * np.zeros((2, 1))
            R0[0,0] = 1e-5
            R0[1,0] = 1e-5
            #self.kf.set_dynamics(A=A0, C=C0, Q=10*Q0, R=20*R0) #6
            self.kf.set_dynamics(A=A0, C=C0, Q=1*Q0, R=0.00001*R0) #6
        else:
            C0 = np.zeros((1,2))
            C0[0, 0] = 1
            C0[0, 1] = 0
            A0 = self.F[0:2,0:2]
            ###############
            Q0 = 4*np.eye(2)#ms.Q
            Q0[1,1] = 1
            #Q0 = ms.Q[0:2,0:2]
            R0 = 1*1e-4*np.ones((1,1))
            mult3 = 0.001
            ###############
            self.kf.set_dynamics(A=A0, C=C0, Q=mult3*Q0, R=mult3*R0) #7

        #self.kf.set_dynamics(A=ms.F, C=torch.eye(ms.F.shape[0]), Q=ms.Q, R=(ms.nof_parts)*ms.Q) #2
        #self.kf.make_velocity_kalman_gain_offline_torch(opt, device)
        #print("updating velocities" if self.update_velocities_kalman else "not updating velociteis")
    def reset_before_batch(self, train_nn, x):
        train_nn1, train_nn2, train_nn3 = train_nn
        self.kf.reset(self.opt, x, self.device)
        self.mm.reset(self.opt, x, self.device)
        self.sm.reset(self.opt, x, self.device)

    def print0(self, new_particles, new_weights, to_be_advenced_particles_ancestors, meas_part_of_initial_sampling, parents_indces):
        print("new_particles: " + str(new_particles))
        print("new_weights: " + str(new_weights))
        print("to_be_advenced_particles_ancestors: " + str(to_be_advenced_particles_ancestors))
        print("meas_part_of_initial_sampling: " + str(meas_part_of_initial_sampling))
        print("parents_indces: " + str(parents_indces))

    def random_uniform_range(self, lower, upper, N, nof_targets):
        out = np.zeros((N, nof_targets, lower.shape[0]))
        for i in np.arange(lower.shape[0]):
            out[:, :, i] = np.random.uniform(low=lower[i], high=upper[i], size=[N, nof_targets]).reshape((N, nof_targets))
        return out

    def create_initial_estimate(self, x0, v0, N):
        x0 = x0.cpu().detach().numpy()
        v0 = v0.cpu().detach().numpy()
        batch_size, nof_targets, _ = x0.shape
        out = np.zeros((batch_size, N, nof_targets,4))
        for traj in np.arange(batch_size):
            max_speed = np.power(self.opt.sig_u,2)*self.opt.tau*self.opt.nof_steps
            lower = np.array([self.sensor_params.center[0] -self.sensor_params.sensor_size[0]/2, -max_speed,self.sensor_params.center[1]-self.sensor_params.sensor_size[1]/2 , -max_speed])
            upper = np.array([self.sensor_params.center[0] + self.sensor_params.sensor_size[0]/2, max_speed, self.sensor_params.center[1]+self.sensor_params.sensor_size[1]/2, max_speed])
            out[traj] = self.random_uniform_range(lower, upper, N, self.opt.nof_targs)
            if self.opt.cheat_first_particles:
                if False:
                    margin = 2
                    little_offset = 0
                    for target_idx in np.arange(out[traj].shape[1]):
                        lower[[0, 2]] = (x0[traj, target_idx, 0]+little_offset - margin), (x0[traj,target_idx, 1]+little_offset - margin)
                        upper[[0, 2]] = (x0[traj, target_idx, 0]+little_offset + margin), (x0[traj,target_idx, 1]+little_offset + margin)
                        out[traj,:, target_idx] =  np.reshape(self.random_uniform_range(lower, upper, N, 1),(N, 4))
                    radius = margin
                else:
                    if not self.opt.cheat_first_vels:
                        #as was trained
                        v0 = 0*v0
                        mult = 100
                    else:
                        # as in paper
                        mult = 1
                    out[traj] = np.tile(np.concatenate((x0[traj:traj + 1, :, 0:1], v0[traj:traj + 1, :, 0:1], x0[traj:traj + 1, :, 1:2], v0[traj:traj + 1, :, 1:2]), axis=-1), (N, 1, 1))
                    out[traj] += np.random.multivariate_normal(np.zeros((4)),self.mm_params.Q*mult,out[traj].shape[:-1])
                    if self.opt.cheat_first_locs_only_half_cheat:
                        out[traj][:, :, (0, 2)] += np.random.multivariate_normal(np.zeros((2)), self.opt.locs_half_cheat_var * np.eye(2), out[traj][:, :, (0, 2)].shape[:-1])
        return out


    def forward(self, prts_locs, prts_vels, ln_weights, parents_incs, z_for_meas, ts_idx, true_vels, true_locs, force_dont_sample_s1=False):
        #particles = particles.to(device)
        #ln_weights = ln_weights.to(device)
        #parents_incs = parents_incs.to(device)
        #z_for_meas = z_for_meas.to(device)
        is_first_sample_mu = False
        batch_size, nof_parts, nof_targs, _ = prts_locs.shape
        state_vector_dim = 4
        particles = torch.zeros((batch_size, nof_parts, nof_targs, state_vector_dim), device=prts_locs.device)
        particles[:, :, :, (0, 2)] = prts_locs
        particles[:, :, :, (1, 3)] = prts_vels
        to_be_advenced_particles_ancestors = copy.deepcopy(particles.detach()).detach()
        #weights = torch.exp(ln_weights)
        #weights = weights / torch.tile(torch.reshape(torch.sum(weights, dim=1), (batch_size, 1)), (1, nof_parts))
        args = (prts_locs, prts_vels, ln_weights, z_for_meas, is_first_sample_mu, self.is_TRAPP, torch.tensor(self.opt.mm_params.F).to(self.device), torch.tensor(self.opt.mm_params.Q).to(self.device), ts_idx)

        atrapp_time1 = time.time()
        new_prts_locs,new_prts_vels, ln_new_weights, parents_indces, intermediates,  timings = atrapp.update_atrapp_torch(opt=self.opt  , args = args         ,
                                                                                                                mm = self.mm  , sm = self.sm, device = self.device,
                                                                                                                records=None  , force_dont_sample_s1=force_dont_sample_s1)
        atrapp_time1 = time.time()-atrapp_time1
        if ms_do_debug_prints:
            print("after update_atrapp")
            self.print0(new_particles, ln_new_weights, None, parents_indces)
        # updating velocities using kalman filer
        curr_step_true_vel = None
        if ts_idx >= 3: # y[0]=None, y[1,2]!=None true_vels[0]=(y[2]-y[1])/tau
            curr_step_true_vel = true_vels[:,ts_idx-2]
        new_particles = torch.zeros((*new_prts_locs.shape[:-1],4),device=new_prts_locs.device)
        new_particles[:,:,:,(0,2)] = new_prts_locs
        new_particles[:,:,:,(1,3)] = new_prts_vels
        new_particles= new_particles.detach()
        new_particles.requires_grad = False
        kf_time1 = time.time()
        new_prts_vels = self.kf.update_particles_velocities8_torch(self.opt, prts_locs.detach(), new_prts_locs.detach(), prts_vels.detach(), parents_indces, self.opt.tau, ts_idx, curr_step_true_vel, self.device)
        kf_time1 = time.time()-kf_time1

        #self.kf.update_particles_velocities7_torch(self.opt, new_particles, to_be_advenced_particles_ancestors, parents_indces, self.opt.tau, curr_step_true_vel, self.device)
        #new_prts_vels = new_particles[:,:,:,(1,3)]
        #print("atrapp_time1: "+str(atrapp_time1)+", kf_time1: "+str(+ kf_time1))
        #t1,t2,t3 = timings
        #timings = atrapp_time1+kf_time1, t2, t3
        if ms_do_debug_prints:
            print("after update_particles_velocities")
            self.print0(new_particles, ln_new_weights, to_be_advenced_particles_ancestors, parents_indces)
        return new_prts_locs,new_prts_vels, ln_new_weights, parents_indces, intermediates, timings

