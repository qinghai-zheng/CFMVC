import h5py

import opt
import random
import numpy as np
import scipy.io as scio
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.cluster import KMeans



def setup():
    """
    setup
    - name: the name of dataset
    - device: CPU / GPU
    - seed: random seed
    - n_clusters: num of cluster
    - n_input: dimension of feature
    - lr: learning rate
    Return: None

    """
    print("setting:")
    setup_seed(opt.args.seed)

    if opt.args.name == 'BBCSport':
        opt.args.n_clusters = 5
        opt.args.n_input1 = 3183
        opt.args.n_input2 = 3203
        opt.args.lr = 1e-3
        opt.args.pretrain_path1 = 'data/BBCSport1.pkl'
        opt.args.pretrain_path2 = 'data/BBCSport2.pkl'



    opt.args.device = torch.device("cuda" if opt.args.cuda else "cpu")
    print("------------------------------")
    print(opt.args)
    # print("dataset       : {}".format(opt.args.name))
    # print("device        : {}".format(opt.args.device))
    # print("random seed   : {}".format(opt.args.seed))
    # print("clusters      : {}".format(opt.args.n_clusters))
    # print("learning rate : {:.0e}".format(opt.args.lr))
    print(opt.args.name)
    # print(opt.args.sigma)
    print(opt.args.p1)
    print(opt.args.p2)
    print("------------------------------")


class load_data(Dataset):
    def __init__(self, dataset):

        if dataset == 'BBCSport':
            x1 = scio.loadmat('data/{}.mat'.format(dataset))['X1']
            x2 = scio.loadmat('data/{}.mat'.format(dataset))['X2']
            y = scio.loadmat('data/{}.mat'.format(dataset))['gt']
            self.x1 = np.transpose(x1)
            self.x2 = np.transpose(x2)
            self.y = np.squeeze(y)


    def __len__(self):
        # 样本数，既矩阵行数
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x1[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))


def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def load_graph(dataset, k, m):

    path = 'graph/{}{}{}_graph.txt'.format(dataset, k, m)
    if dataset == 'BBCSport':
        n = 544

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=None, device='cuda', requires_grad=False)

def numpy_to_torch(a, sparse=False):
    """
    numpy array to torch tensor
    :param a: the numpy array
    :param sparse: is sparse tensor or not
    :return: torch tensor
    """
    if sparse:
        a = torch.sparse.Tensor(a)
        a = a.to_sparse()
    else:
        a = torch.FloatTensor(a)
    return a



def target_distribution(q):
    # return p
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def off_diagonal(x):
    """
    off-diagonal elements of x
    Args:
        x: the input matrix
    Returns: the off-diagonal elements of x
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()




def dicr_loss(z1, z2):
    """
    Dual Information Correlation Reduction loss L_{DICR}
    Args:
        Z_ae: AE embedding including two-view node embedding [0, 1] and two-view cluster-level embedding [2, 3]
        Z_igae: IGAE embedding including two-view node embedding [0, 1] and two-view cluster-level embedding [2, 3]
    Returns:
        L_{DICR}
    """
    # Sample-level Correlation Reduction (SCR)
    # cross-view sample correlation matrix
    # C = cross_correlation(z1, z2)
    C = torch.mm(F.normalize(z1, dim=1), F.normalize(z2, dim=1).t())
    loss_dicr = torch.diagonal(C).add(-1).pow(2).mean() + off_diagonal(C).pow(2).mean()
    return loss_dicr




def model_init(model,data1,data2,adj1,adj2):

    with torch.no_grad():
        _, _, _, z, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(data1, data2, adj1, adj2)

    kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(opt.args.device)
