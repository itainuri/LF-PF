import random
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal as torch_mulvar_norm

class MotionModelParams:
    def __init__(self,
                 tau=1,
                 sig_u = 0.1
                 ):
        super().__init__()
        self.tau = tau
        self.sig_u = sig_u
        self.sig_u2 = np.power(self.sig_u, 2)

        self.F = np.kron(np.eye(2), [[1, self.tau], [0, 1]])
        Q_tag = [[np.power(self.tau, 3) / 3, np.power(tau, 2) / 2], [np.power(self.tau, 2) / 2, self.tau]]
        self.Q = self.sig_u2 * np.kron(np.eye(2), Q_tag)
        #self.Qs = np.tile(Q, (nof_targets, 1, 1))
        #self.Fs = np.tile(F, (nof_targets, 1, 1))


class MotionModel(object):
    def __init__(self, opt, device):
        self.device = device
        self.opt = opt
        #self.reset(opt, device)
    def reset(self, opt, x, device):
        self.opt = opt # TODO check if should be self
        meas_cov = torch.zeros((opt.state_vector_dim, opt.state_vector_dim), device=device)
        meas_cov[0, 0] = 1
        meas_cov[2, 2] = 1
        meas_cov = 0.1 * meas_cov
        #meas_cov = 0.05* meas_cov
        if opt.cheat_first_vels:
            meas_cov = 0 * meas_cov
        Q_torch = torch.tensor(opt.mm_params.Q, device=device)
        #batch_size = np.maximum(self.opt.batch_size, self.opt.batch_size_val)
        batch_size = x.shape[0]
        locs = torch.tile(torch.zeros((1, 1, 1, 4),device=device), (batch_size, self.opt.nof_parts, self.opt.nof_targs, 1))
        covs = torch.tile(torch.reshape(Q_torch + meas_cov, list((1, 1, 1, * Q_torch.shape))), (batch_size, self.opt.nof_parts, self.opt.nof_targs, 1, 1)).to(device)
        self.torch_noises = torch_mulvar_norm(loc=locs, covariance_matrix=covs)

        self.F_locs = torch.zeros((2, 2), device=device)
        self.F_locs_from_vels = torch.zeros((2, 2), device=device)
        self.F_vels = torch.zeros((2, 2), device=device)
        self.F_locs[0, 0] = opt.mm_params.F[0, 0]
        self.F_locs[1, 1] = opt.mm_params.F[2, 2]
        self.F_locs_from_vels[0, 0] = opt.mm_params.F[0, 1]
        self.F_locs_from_vels[1, 1] = opt.mm_params.F[2, 3]
        self.F_vels[0, 0] = opt.mm_params.F[1, 1]
        self.F_vels[1, 1] = opt.mm_params.F[3, 3]

        self.Qp = torch.tensor([[opt.mm_params.Q[0, 0], opt.mm_params.Q[0, 2]], [opt.mm_params.Q[2, 0], opt.mm_params.Q[2, 2]]])
        self.Qv = torch.tensor([[opt.mm_params.Q[1, 1], opt.mm_params.Q[1, 3]], [opt.mm_params.Q[3, 1], opt.mm_params.Q[3, 3]]])
        self.Qvp = torch.tensor([[opt.mm_params.Q[1, 0], opt.mm_params.Q[1, 2]], [opt.mm_params.Q[3, 0], opt.mm_params.Q[3, 2]]])

        self.chol_Qp = torch.transpose(torch.cholesky(self.Qp),0,1)
        self.chol_Qv = torch.transpose(torch.cholesky(self.Qv),0,1)
        loc_noise_mult = 1
        vel_noise_mult = 1
        #loc_noise_mult = 1
        #vel_noise_mult = 1
        self.mult_mat = torch.tensor([loc_noise_mult, vel_noise_mult, loc_noise_mult, vel_noise_mult], device=device)

    def get_particles_noise(self, is_mu, nof_batches, nof_parts, nof_targs):
        if is_mu:
            noises_resampled = torch.zeros((nof_batches, nof_parts, nof_targs,4), device=self.torch_noises.mean.device)
        else:
            noises_resampled = self.torch_noises.rsample()
        curr_noise = torch.multiply(noises_resampled[:nof_batches, :nof_parts, :nof_targs, :], self.mult_mat)
        curr_noise = curr_noise.detach()
        curr_noise.requires_grad = False
        return curr_noise

    def advance_locations(self, opt, is_mu, old_prts_locs, old_prts_vels, device, print_seed = False, print_grad = True):
        batch_size, nof_parts, nof_targs, state_vector_loc_dim = old_prts_locs.shape
        assert len(old_prts_locs.shape)==4



       # synchronizez seed for when this wasnt a class
        rand_int = torch.randint(10000000000,(1,))
        if print_seed:
            print("seed: ")
            torch_rand_test = torch.randn(1)
            print(torch_rand_test)
            numpy_rand_test = np.random.randn(1)
            print(numpy_rand_test)
            python_rand_test = random.random()
            print(python_rand_test)
            print(noises_resampled)
            print(rand_int)

        torch.manual_seed(rand_int)
        nof_batches, nof_parts, nof_targs, _ = old_prts_locs.shape

        curr_noise = self.get_particles_noise(is_mu, nof_batches, nof_parts, nof_targs)

        #new_particles = torch.transpose(torch.matmul(F, torch.transpose(old_particles, 1, 2)), 1, 2) + 5*noises_resampled[:, :old_particles.shape[1], :]
        if 1:
            #new_particles = torch.transpose(torch.matmul(F, torch.transpose(old_particles, 2, 3)), 2, 3) + curr_noise[:batch_size]
            #TODO check that curr_noise.shape_nof_parts is not 1
            #F_vels = [[F[0,0] , 0],[0, F[2,2]]]
            new_prts_locs = torch.matmul(old_prts_locs, torch.transpose(self.F_locs,0,1)) + torch.matmul(old_prts_vels, torch.transpose(self.F_locs_from_vels,0,1)) +curr_noise[:,:,:,(0,2)]
            new_prts_vels = torch.matmul(old_prts_vels, torch.transpose(self.F_vels,0,1)) +curr_noise[:,:,:,(1,3)]

        return new_prts_locs, new_prts_vels
