import torch
import torch.nn as nn

dim0=32

class PRT_EMB3(nn.Module):
    def __init__(self, nof_parts=100, embed_dim=128, is_wt_per_targ = False):
        super(PRT_EMB3, self).__init__()
        self.nof_parts = nof_parts
        qq_in = 3
        out_features = embed_dim

        fc1_in_channels  = qq_in
        fc2_in_channels  = embed_dim
        fc3_in_channels  = embed_dim
        fc4_in_channels  = embed_dim
        fc5_in_channels  = embed_dim

        fc1_out_channels = fc2_in_channels
        fc2_out_channels = fc3_in_channels
        fc3_out_channels = fc4_in_channels
        fc4_out_channels = fc5_in_channels
        fc5_out_channels = out_features

        self.fc1 = nn.Conv2d(in_channels=fc1_in_channels, out_channels=fc1_out_channels, kernel_size=1, groups=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=fc2_in_channels, out_channels=fc2_out_channels, kernel_size=1, groups=1, bias=True)
        self.fc3 = nn.Conv2d(in_channels=fc3_in_channels, out_channels=fc3_out_channels, kernel_size=1, groups=1, bias=True)
        self.fc4 = nn.Conv2d(in_channels=fc4_in_channels, out_channels=fc4_out_channels, kernel_size=1, groups=1, bias=True)
        self.fc5 = nn.Conv2d(in_channels=fc5_in_channels, out_channels=fc5_out_channels, kernel_size=1, groups=1, bias=True)

        self.activation = torch.nn.LeakyReLU()

    def forward(self, x0):
        x = self.activation(self.fc1(x0))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)
        return x

class PRT_EMB12(nn.Module):
    def __init__(self, nof_parts=100, embed_dim_per_group=128, is_wt_per_targ = False, fc_groups=1):
        super(PRT_EMB12, self).__init__()
        self.nof_parts = nof_parts
        qq_in = (2 + 1) if not is_wt_per_targ else 3
        out_features = embed_dim_per_group*2

        fc1_in_channels  = qq_in*2
        fc2_in_channels  = embed_dim_per_group*2
        fc3_in_channels  = embed_dim_per_group*2
        fc4_in_channels  = embed_dim_per_group*2
        fc5_in_channels  = embed_dim_per_group*2

        fc1_out_channels = fc2_in_channels
        fc2_out_channels = fc3_in_channels
        fc3_out_channels = fc4_in_channels
        fc4_out_channels = fc5_in_channels
        fc5_out_channels = out_features

        self.fc1 = nn.Conv2d(in_channels=fc1_in_channels, out_channels=fc1_out_channels, kernel_size=1, groups=2, bias=True)
        self.fc2 = nn.Conv2d(in_channels=fc2_in_channels, out_channels=fc2_out_channels, kernel_size=1, groups=2, bias=True)
        self.fc3 = nn.Conv2d(in_channels=fc3_in_channels, out_channels=fc3_out_channels, kernel_size=1, groups=2, bias=True)
        self.fc4 = nn.Conv2d(in_channels=fc4_in_channels, out_channels=fc4_out_channels, kernel_size=1, groups=2, bias=True)
        self.fc5 = nn.Conv2d(in_channels=fc5_in_channels, out_channels=fc5_out_channels, kernel_size=1, groups=2, bias=True)

        self.activation = torch.nn.LeakyReLU()

    def forward(self, x0):
        assert not torch.isnan(self.fc1.weight[0,0])
        x = self.activation(self.fc1(x0))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)
        return x

class NET1_QKV9(nn.Module):
    def __init__(self, nof_parts, head_in_dim, head_out_dim, nof_types):
        super(NET1_QKV9, self).__init__()
        fc1_in_channels  = 3*head_in_dim
        fc1_out_channels = 3*head_out_dim
        self.fc1 = nn.Conv2d(in_channels=fc1_in_channels, out_channels=fc1_out_channels, kernel_size=1, groups=3, bias=True)

    def forward(self, x0):
        batch_size = x0.shape[0]
        x = torch.tile(x0, (1, 3,1,1))
        x = self.fc1(x)
        return x

class NET1_QKV12(nn.Module):
    def __init__(self, nof_parts, head_in_dim, head_out_dim, qkv_nof_types):
        super(NET1_QKV12, self).__init__()
        head_in_dim = head_in_dim
        self.qkv_nof_types = qkv_nof_types
        fc1_in_channels  = 6*head_in_dim
        fc1_out_channels = 6*head_out_dim
        self.fc1 = nn.Conv2d(in_channels=fc1_in_channels, out_channels=fc1_out_channels, kernel_size=1, groups=6, bias=True)

    def forward(self, x0):
        batch_size = x0.shape[0]
        x = torch.tile(x0, (1, self.qkv_nof_types,1,1))
        x = self.fc1(x)
        return x

class SA1_009(nn.Module):
    def __init__(self, nof_parts=100,skip=False,particle_embed_dim=dim0, q_head_out_dim=dim0, v_head_out_dim=dim0):
        super(SA1_009, self).__init__()
        self.nof_parts = nof_parts
        self.skip = skip
        self.particle_embed_dim = particle_embed_dim
        self.nof_types = 3
        self.q_head_in_dim = self.particle_embed_dim  # int(particle_embed_dim)
        self.k_head_in_dim = self.q_head_in_dim
        self.v_head_in_dim = self.particle_embed_dim
        self.emb_in_dim = 3
        self.v_head_out_dim = v_head_out_dim  # int(particle_embed_dim)
        self.q_head_out_dim = q_head_out_dim
        self.k_head_out_dim = self.q_head_out_dim
        self.out_dim_wts = 1
        self.out_dim_locs = 2
        self.W_h1_in_per_group = self.particle_embed_dim + self.v_head_out_dim
        self.qkv_nof_types = 3
        self.particle_emb = PRT_EMB3(nof_parts=self.nof_parts, embed_dim=self.particle_embed_dim, is_wt_per_targ=1)
        self.net_qkv = NET1_QKV9(nof_parts=self.nof_parts,head_in_dim=self.q_head_in_dim, head_out_dim=self.q_head_out_dim, nof_types=self.qkv_nof_types)
        #for weights
        self.W_h1 = nn.Conv2d(in_channels=self.W_h1_in_per_group, out_channels=self.v_head_out_dim, kernel_size=1, groups=1, bias=True)
        self.W_h1_0 = nn.Conv2d(in_channels=self.v_head_out_dim, out_channels=self.v_head_out_dim, kernel_size=1, groups=1, bias=True)
        self.W_h1_1 = nn.Conv2d(in_channels=self.v_head_out_dim, out_channels=self.v_head_out_dim, kernel_size=1, groups=1, bias=True)
        self.W_h2 = nn.Conv2d(in_channels=self.v_head_out_dim, out_channels= self.out_dim_wts, kernel_size=1, groups=1, bias=True)
        #for locations
        self.W_h3 = nn.Conv2d(in_channels=self.W_h1_in_per_group, out_channels=self.v_head_out_dim, kernel_size=1, groups=1, bias=True)
        self.W_h3_0 = nn.Conv2d(in_channels=self.v_head_out_dim, out_channels=self.v_head_out_dim, kernel_size=1, groups=1, bias=True)
        self.W_h3_1 = nn.Conv2d(in_channels=self.v_head_out_dim, out_channels=self.v_head_out_dim, kernel_size=1, groups=1, bias=True)
        self.W_h4 = nn.Conv2d(in_channels=self.v_head_out_dim, out_channels= self.out_dim_locs, kernel_size=1, groups=1, bias=True)
        if not self.skip:
            assert self.particle_embed_dim % self.q_head_in_dim == 0, "self.particle_embed_dim % self.q_head_in_dim == 0"
        self.activation = torch.nn.LeakyReLU()
        self.activation_wts = torch.nn.LeakyReLU()

    def forward(self, x0_0):
        batch_size, nof_parts, in_dim = x0_0.shape
        if self.skip:
            return x0_0
        else:
            x0=x0_0
            x0[:, :, -1] = 1 / x0[:, :, -1]
        x = torch.tile(x0, (1, 1, 1))
        x = torch.reshape(x, (batch_size * self.nof_parts, self.emb_in_dim, 1, 1))
        x_e = self.particle_emb(x)
        qkv = self.net_qkv(x_e)
        qkv = torch.reshape(qkv,(batch_size, nof_parts, self.nof_types, self.particle_embed_dim))
        per_type_width = 1
        qs = qkv[:, :, :per_type_width, :]
        ks = qkv[:, :, per_type_width:2 * per_type_width, :]
        vs = qkv[:, :, 2 * per_type_width:, :]
        q = torch.reshape(torch.permute(qs, (0, 2, 1, 3)), (batch_size * per_type_width, self.nof_parts, self.q_head_out_dim))
        k = ks / torch.sqrt(torch.tensor(self.k_head_out_dim))  # (bs, n_heads, q_length, dim_per_head)
        k = torch.reshape(torch.permute(k, (0, 2, 3, 1)), (batch_size * per_type_width, self.k_head_out_dim, self.nof_parts))
        ip = torch.bmm(q, k)
        sip = torch.softmax(ip, dim=-1)
        v = torch.reshape(torch.permute(vs, (0, 2, 1, 3)), (batch_size * per_type_width, self.nof_parts, self.v_head_out_dim))
        x = torch.bmm(sip, v)
        x = x.reshape((batch_size, 1, self.nof_parts, self.v_head_out_dim))
        x_e = torch.reshape(x_e,(batch_size, self.nof_parts, 1,self.particle_embed_dim))
        x_e = torch.permute(x_e,(0, 2, 1, 3))
        v_before = torch.permute(vs, (0, 2, 1, 3))
        x_xe = torch.cat((v_before, x), dim=3)
        x_xe = torch.permute(x_xe, (0, 2, 1, 3))
        x_xe = torch.reshape(x_xe, (batch_size * self.nof_parts, self.W_h1_in_per_group, 1, 1))
        x = self.W_h1(x_xe)
        x = self.activation_wts(x)
        x = self.activation_wts(self.W_h1_0(x))
        x = self.activation_wts(self.W_h1_1(x))
        x = self.W_h2(x)
        x2 = self.W_h3(x_xe)  # (bs, q_length, dim)
        x2 = self.activation_wts(x2)
        x2 = self.activation_wts(self.W_h3_0(x2))
        x2 = self.activation_wts(self.W_h3_1(x2))
        x2 = self.W_h4(x2)
        correction = torch.cat((x2.reshape(batch_size, self.nof_parts, self.out_dim_locs), x.reshape(batch_size, self.nof_parts, self.out_dim_wts)), dim=-1)
        x = x0_0 + correction
        x[:, :, -1] = 1 / x[:, :, -1]
        return x

class SDP12(nn.Module):
    def __init__(self, nof_parts=100, particle_embed_dim=dim0, q_head_out_dim=dim0, v_head_out_dim=dim0):
        super(SDP12, self).__init__()
        self.nof_parts = nof_parts
        self.particle_embed_dim = particle_embed_dim
        self.output_wts = 1
        self.output_locs = 1
        self.wts_and_or_locs = self.output_wts+self.output_locs
        self.q_head_in_dim = self.particle_embed_dim  # int(particle_embed_dim)
        self.k_head_in_dim = self.q_head_in_dim
        self.v_head_in_dim = self.particle_embed_dim
        self.fc_groups_per_network = 2
        self.v_head_out_dim = v_head_out_dim  # int(particle_embed_dim)
        self.q_head_out_dim = q_head_out_dim
        self.k_head_out_dim = self.q_head_out_dim
        self.first_linear_out_dim=v_head_out_dim
        self.W_h1_in_per_group = self.particle_embed_dim + self.v_head_out_dim
        self.qkv_nof_types = 3
        self.net_qkv = NET1_QKV12(nof_parts=self.nof_parts, head_in_dim=self.q_head_in_dim, head_out_dim=self.q_head_out_dim, qkv_nof_types=self.qkv_nof_types)
        self.W_h1 = nn.Conv2d(in_channels=self.W_h1_in_per_group * self.fc_groups_per_network, out_channels=self.fc_groups_per_network * self.v_head_out_dim, kernel_size=1, groups=2, bias=True)
        self.W_h1_0 = nn.Conv2d(in_channels=self.v_head_out_dim * self.fc_groups_per_network, out_channels=self.fc_groups_per_network * self.v_head_out_dim, kernel_size=1, groups=2, bias=True)
        self.W_h1_1 = nn.Conv2d(in_channels=self.v_head_out_dim * self.fc_groups_per_network, out_channels=self.fc_groups_per_network * self.v_head_out_dim, kernel_size=1, groups=2, bias=True)
        assert self.particle_embed_dim % self.q_head_in_dim == 0, "self.particle_embed_dim % self.q_head_in_dim ==0"
        self.activation_wts = torch.nn.LeakyReLU()

    def forward(self, x_e):
        batch_size, nof_parts, in_dim = x_e.shape
        x_e = torch.reshape(x_e, (batch_size * self.nof_parts, in_dim, 1, 1))
        qkv = self.net_qkv(x_e)
        qkv2 = torch.reshape(qkv, (batch_size, nof_parts, self.qkv_nof_types, 1, 2, self.particle_embed_dim))
        qkv3 = torch.permute(qkv2, (0, 4, 2, 3, 1, 5))
        qs = qkv3[:, :, 0]
        ks = qkv3[:, :, 1]
        vs = qkv3[:, :, 2]
        q = torch.reshape(qs, (batch_size * 2 , nof_parts, self.q_head_out_dim))
        k = torch.reshape(ks, (batch_size * 2, nof_parts, self.q_head_out_dim))
        k = torch.permute(k, (0, 2, 1)) / torch.sqrt(torch.tensor(self.k_head_out_dim))  # (bs, n_heads, q_length, dim_per_head)
        ip = torch.bmm(q, k)
        sip = torch.softmax(ip / torch.sqrt(torch.tensor(self.k_head_out_dim)), dim=-1)
        v = torch.reshape(vs, (batch_size * 2, nof_parts, self.q_head_out_dim))
        x = torch.bmm(sip, v)
        x = x.reshape((batch_size, 2 , nof_parts, self.v_head_out_dim))
        v_before = torch.reshape(vs, (batch_size, 2 , nof_parts, self.v_head_out_dim))
        x_xe = torch.cat((v_before, x), dim=3)
        x_xe = torch.permute(x_xe, (0, 2, 1, 3))  # the order 2,1 and not 1,2 to get the different heads of the same target to be close for the conv2 groups
        x_xe = torch.reshape(x_xe, (batch_size* self.nof_parts , 2*self.W_h1_in_per_group, 1, 1))
        x = self.W_h1(x_xe)
        x = self.activation_wts(x)
        x = self.activation_wts(self.W_h1_0(x))
        x = self.activation_wts(self.W_h1_1(x))
        return x

class SA1_012(nn.Module):
    def __init__(self, nof_parts=100, skip=False, particle_embed_dim=dim0, q_head_out_dim=dim0, v_head_out_dim=dim0, another_sdp=True):
        super(SA1_012, self).__init__()
        self.another_sdp = another_sdp
        self.nof_parts = nof_parts
        self.skip = skip
        self.particle_embed_dim = particle_embed_dim
        self.output_wts = 1
        self.output_locs = 1
        self.wts_and_or_locs = self.output_wts+self.output_locs
        self.q_head_in_dim = self.particle_embed_dim  # int(particle_embed_dim)
        self.k_head_in_dim = self.q_head_in_dim
        self.v_head_in_dim = self.particle_embed_dim
        self.emb_in_dim = 3
        self.v_head_out_dim = v_head_out_dim  # int(particle_embed_dim)
        self.q_head_out_dim = q_head_out_dim
        self.k_head_out_dim = self.q_head_out_dim
        self.out_dim_wts = 1
        self.first_linear_out_dim=v_head_out_dim
        self.W_h5_output_dim = 2
        self.W_h4_output_dim = 1
        self.post_sdp_groups = 2
        self.out_dim_locs = 2
        self.W_h1_in_per_group = (self.particle_embed_dim + self.v_head_out_dim)
        SDP = SDP12
        self.particle_emb = PRT_EMB12(nof_parts=self.nof_parts,embed_dim_per_group=self.particle_embed_dim, is_wt_per_targ=0, fc_groups=2)
        self.sdp1 = SDP(nof_parts=nof_parts, particle_embed_dim=particle_embed_dim, q_head_out_dim=q_head_out_dim, v_head_out_dim=v_head_out_dim)
        if self.another_sdp:
            self.W_h2 = nn.Conv2d(in_channels=2*self.v_head_out_dim, out_channels= 2*self.first_linear_out_dim, kernel_size=1, groups=2, bias=True)
            self.sdp2 = SDP(nof_parts=nof_parts, particle_embed_dim=particle_embed_dim, q_head_out_dim=q_head_out_dim, v_head_out_dim=v_head_out_dim)
        self.W_h3 = nn.Conv2d(in_channels=self.v_head_out_dim, out_channels=self.first_linear_out_dim, kernel_size=1, groups=1, bias=True)
        self.W_h4 = nn.Conv2d(in_channels=self.first_linear_out_dim, out_channels=self.out_dim_locs+self.out_dim_wts, kernel_size=1, groups=1, bias=True)
        self.W_h5 = nn.Conv2d(in_channels=self.v_head_out_dim, out_channels=self.first_linear_out_dim, kernel_size=1, groups=1, bias=True)
        self.W_h6 = nn.Conv2d(in_channels=self.first_linear_out_dim, out_channels=self.out_dim_locs+self.out_dim_wts, kernel_size=1, groups=1, bias=True)
        self.W_h4_1 = nn.Conv2d(in_channels=self.W_h4_output_dim * 2, out_channels= self.W_h4_output_dim*2, kernel_size=1, groups=2, bias=True)
        self.W_h4_2 = nn.Conv2d(in_channels=self.W_h4_output_dim * 2, out_channels= self.W_h4_output_dim*2, kernel_size=1, groups=2, bias=True)
        if not self.skip:
            assert self.particle_embed_dim % self.q_head_in_dim == 0, "d_model % should be zero."
        self.activation_wts = torch.nn.LeakyReLU()

    def forward(self, x0_0):
        batch_size, nof_parts, in_dim = x0_0.shape
        if self.skip:
            return x0_0
        else:
            x0=x0_0
            x0[:, :, -1] = 1/x0[:, :, -1]
        x = torch.tile(x0, (1, 1, 2))
        x = torch.reshape(x, (batch_size * self.nof_parts, self.emb_in_dim*2, 1, 1))
        x_e = self.particle_emb(x)
        x_e = torch.reshape(x_e, (batch_size, self.nof_parts, self.particle_embed_dim*2))
        x = self.sdp1(x_e)
        if self.another_sdp:
            x = self.W_h2(x)
            x = torch.reshape(x, (batch_size, self.nof_parts, 2*self.particle_embed_dim))
            x = self.sdp2(x)
        x1 = self.activation_wts(self.W_h3(x[:, :self.first_linear_out_dim]))
        x1 = self.activation_wts(self.W_h4(x1))
        x2 = torch.mean(torch.reshape(x[:, self.first_linear_out_dim:], (batch_size, self.nof_parts, self.particle_embed_dim)), dim=1)
        x2 = torch.reshape(x2, (*x2.shape, 1, 1))
        x2 = self.activation_wts(self.W_h5(x2))
        x2 = self.activation_wts(self.W_h6(x2))
        x = x0_0 + torch.reshape(x1, x0_0.shape)
        x[:, :, :-1] += torch.reshape(x2, (batch_size, 1, x0_0.shape[-1]))[:, :, :-1]
        x[:, :, -1] = 1 / x[:, :, -1]
        return x