import copy

import numpy as np
import math
import scipy.linalg

from builtins import range

import torch


class KalmanFilter(object):
    def __init__(self, A=None, C=None, Q=None, R=None, device=None):

        self.A = None
        self.C = None
        self.R = None  # Measurement noise covariance
        self.Q = None  # Process noise covariance
        self.set_dynamics(A, C, Q, R)
        self.P = None
        self.P_x = None
        self.P_y = None
        self.device = device
        self.debug_count = 0

    def reset(self, opt, x, device):
        self.P = None
        self.xkm1_avg = None
        self.P_x = None
        self.P_y = None
        self.kf_parts = None
        self.make_velocity_kalman_gain_offline_torch(opt, x, device)


    def set_dynamics(self, A=None, C=None, Q=None, R=None, f_k=None, h_k=None):
        if (A is not None):
            self.A = A
        if (C is not None):
            self.C = C
        if (Q is not None):
            self.Q = Q
        if (R is not None):
            self.R = R
            self.xkm1_avg = None
        if 1:
            if (self.A is not None):
                self.A = torch.tensor(A).to(self.device)
            if (self.C is not None):
                self.C = torch.tensor(C).to(self.device)
            if (self.Q is not None):
                self.Q = torch.tensor(Q).to(self.device)
            if (self.R is not None):
                self.R = torch.tensor(R).to(self.device)


    def make_velocity_kalman_gain_offline_torch(self, opt, x, device):
        #nof_steps = np.maximum(opt.nof_steps, opt.nof_steps_val)
        nof_steps = x.shape[1]
        velocity_Kalman_covariance = torch.zeros(nof_steps, 2, 2)
        velocity_Kalman_covariance[0] = 10 * torch.eye(2)
        velocity_Kalman_gain = torch.zeros(nof_steps,2, 2)

        Qp = torch.tensor([[opt.mm_params.Q[0, 0], opt.mm_params.Q[0, 2]], [opt.mm_params.Q[2, 0], opt.mm_params.Q[2, 2]]])
        Qv = torch.tensor([[opt.mm_params.Q[1, 1], opt.mm_params.Q[1, 3]], [opt.mm_params.Q[3, 1], opt.mm_params.Q[3, 3]]])
        Qvp = torch.tensor([[opt.mm_params.Q[1, 0], opt.mm_params.Q[1, 2]], [opt.mm_params.Q[3, 0], opt.mm_params.Q[3, 2]]])
        for i in np.arange(nof_steps):
            psi = opt.tau * velocity_Kalman_covariance[i] + Qvp
            S = np.square(opt.tau) * velocity_Kalman_covariance[i] + Qp
            invS = torch.linalg.inv(S)
            cov_qv = velocity_Kalman_covariance[i] + Qv
            velocity_Kalman_gain[i] = psi * invS
            if i == nof_steps-1: break
            velocity_Kalman_covariance[i + 1] = cov_qv - psi * invS * torch.torch.transpose(psi, 0, 1)
        self.Kk = velocity_Kalman_gain.to(device)

    def update_particles_velocities8_torch(self, opt, old_prts_locs, new_prts_locs, old_prts_vels, parents_indcs, tau, curr_ts_idx, true_vel, device):
        # R 1x1
        # Q 2x2
        # C 1x2
        #nof_parts, nof_targs = prts_locs.shape[0:2]
        batch_size, nof_parts, nof_targs, state_vector_loc_dim = old_prts_locs.shape
        batch_indcs = torch.tile(torch.reshape(torch.from_numpy(np.arange(batch_size)).to(device), (batch_size, 1, 1)), (1, nof_parts, nof_targs)).to(torch.long)
        targ_indices = torch.tile(torch.reshape(torch.arange(nof_targs), (1, 1, nof_targs)), (batch_size, nof_parts, 1)).to(device)
        #targ_indices = torch.tile(torch.reshape(torch.arange(nof_targs), (1, nof_targs)), (nof_parts, 1)).to(ms_device)
        sampled_old_parents_locs = old_prts_locs[batch_indcs, parents_indcs, targ_indices]
        sampled_old_parents_vels = old_prts_vels[batch_indcs, parents_indcs, targ_indices]
        try:
            updated_prts_vels = sampled_old_parents_vels + torch.matmul((new_prts_locs - (sampled_old_parents_locs + tau*sampled_old_parents_vels)), self.Kk[curr_ts_idx-1])
        except:
            fdsasf = 9
        #particles_RB(indicesX(b): indicesY(b),:)= particles_RB(indicesX(b): indicesY(b), p2(b,:))+velocity_Kalman_gain(:, 2 * (step - 1) + 1: 2 * (step - 1) + 2)*(particles(indicesX(b):indicesY(b),:)-(previousParticles(indicesX(b):indicesY(b), p2(b,:))+tau * particles_RB(indicesX(b): indicesY(b), p2(b,:))));

        # x{k|k} = x{k|k-1} + Kk*meas_err
        # particles[:, :, (1, 3)] = x_k_km1[:, :, (1, 3)] + torch.matmul(Kk, torch.unsqueeze(meas_err, 3)).squeeze(-1)[:, :, (1, 3)]
        # particles = x_k_km1 + torch.matmul(Kk, torch.unsqueeze(meas_err, 3)).squeeze(-1)
        # meas_err 1x1, Kk 2x1,
        if opt.cheat_get_true_vels and true_vel is not None:
            print(self.debug_count)
            self.debug_count += 1
            if self.debug_count <= opt.cheat_get_true_vels_how_many:
                #updated_prts_vels = torch.from_numpy(true_vel).to(device)
                updated_prts_vels = torch.unsqueeze(torch.from_numpy(true_vel).to(device),1)

            # particles[:, :, (1, 3)] = -0.5
        try:
            return updated_prts_vels
        except:
            dsfsedfs  =9

