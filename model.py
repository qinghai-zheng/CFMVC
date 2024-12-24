from __future__ import print_function, division
import opt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear
from GNN import GNNLayer


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):

        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        h = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(h))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, h


class GAE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_input, n_z, n_clusters):
        super(GAE, self).__init__()
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_z)

    def forward(self, x, adj, adjx, tra1, tra2, tra3, h):
        sigma = opt.args.sigma
        z = self.gnn_1(x, adj)
        lay_1 = z
        # z = self.gnn_1(x, (1 - sigma) * adj + sigma * adjx)
        z = self.gnn_2((1 - sigma) * z + sigma * tra1,  (1 - sigma) * adj + sigma * adjx)
        lay_2 = z
        z = self.gnn_3((1 - sigma) * z + sigma * tra2,  (1 - sigma) * adj + sigma * adjx)
        lay_3 = z
        z = self.gnn_4((1 - sigma) * z + sigma * tra3,  (1 - sigma) * adj + sigma * adjx)
        lay_4 = z
        z = self.gnn_5((1 - sigma) * z + sigma * h, (1 - sigma) * adj + sigma * adjx, active=False)

        return z, lay_1, lay_2, lay_3, lay_4


class CFMVC(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input1, n_input2, n_z, n_clusters, v=1):
        super(CFMVC, self).__init__()

        # autoencoder for intra information
        self.ae1 = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input1,
            n_z=n_z)
        self.ae1.load_state_dict(torch.load(opt.args.pretrain_path1, map_location='cpu'))

        self.ae2 = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input2,
            n_z=n_z)
        self.ae2.load_state_dict(torch.load(opt.args.pretrain_path2, map_location='cpu'))

        # GCN for inter information
        self.gae1 = GAE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_input=n_input1,
            n_z=n_z,
            n_clusters=n_clusters)

        self.gae2 = GAE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_input=n_input2,
            n_z=n_z,
            n_clusters=n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x1, x2, adj1, adj2):
        # node embedding encoded by DNN
        x1_bar, x1_tra1, x1_tra2, x1_tra3, x1_h = self.ae1(x1)
        x2_bar, x2_tra1, x2_tra2, x2_tra3, x2_h = self.ae2(x2)
        h = 0.5 * x1_h + 0.5 * x2_h



        # cross-view information fusion/node embedding encoded by GCN
        z1, z1_lay1, z1_lay2, z1_lay3, z1_lay4 = self.gae1(x1, adj1, adj2, x2_tra1, x2_tra2, x2_tra3, x2_h)
        z2, z2_lay1, z2_lay2, z2_lay3, z2_lay4 = self.gae2(x2, adj2, adj1, x1_tra1, x1_tra2, x1_tra3, x1_h)
        # z1 = F.softmax(z1, dim=1)
        # z2 = F.softmax(z2, dim=1)
        z = 0.5 * z1 + 0.5 * z2

        # # not cross-view information fusion
        # z1 = self.gae1(x1, adj1, adj1, x1_tra1, x1_tra2, x1_tra3, x1_h)
        # z2 = self.gae2(x2, adj2, adj2, x2_tra1, x2_tra2, x2_tra3, x2_h)
        # # z1 = F.softmax(z1, dim=1)
        # # z2 = F.softmax(z2, dim=1)
        # z = 0.5 * z1 + 0.5 * z2

        # the soft assignment distribution Q
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x1_bar, x2_bar, q, z, h, z1, z2, x1_tra1, x1_tra2, x1_tra3, x1_h, x2_tra1, x2_tra2, x2_tra3, x2_h, z1_lay1, z1_lay2, z1_lay3, z1_lay4, z2_lay1, z2_lay2, z2_lay3, z2_lay4


