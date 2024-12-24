import opt
from evaluation import clustering
from utils import *
from model import CFMVC
from torch.optim import Adam
import torch.nn.functional as F


def train(model, data1, data2, adj1, adj2, y):

    print("Training-------------------------------")
    # initialize cluster centers
    model_init(model, data1, data2, adj1, adj2)

    optimizer = Adam(model.parameters(), lr=opt.args.lr)

    for epoch in range(opt.args.epoch):
        x1_bar, x2_bar, q, z, _, z1, z2, _, _, _, _, _, _, _, _,_, _, _, _,_, _, _, _ = model(data1, data2, adj1, adj2)

        tmp_q = q.data
        p = target_distribution(tmp_q)

        re_loss = F.mse_loss(x1_bar, data1) + F.mse_loss(x2_bar, data2)
        di_loss = dicr_loss(z1, z2)       # 平衡性特征融合损失/z1、z2冗余减少
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')

        loss = re_loss + opt.args.p1 * di_loss + opt.args.p2 * kl_loss  #重构损失+平衡性特征融合损失+聚类KL损失
        # loss = re_loss + 10 * kl_loss

        print('{} loss: {}'.format(epoch, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc, nmi, ari, f1 = clustering(z, y)
        if acc > opt.args.acc:
            opt.args.acc = acc
            opt.args.nmi = nmi
            opt.args.ari = ari
            opt.args.f1 = f1




if __name__ == '__main__':
    # setup
    setup()

    # initialize network
    model = CFMVC(128, 256, 512, 512, 256, 128,
                 n_input1=opt.args.n_input1,
                 n_input2=opt.args.n_input2,
                 n_z=opt.args.n_z,
                 n_clusters=opt.args.n_clusters,
                 v=1).to(opt.args.device)
    print(model)

    # load data
    adj1 = load_graph(opt.args.name, opt.args.k, 1)
    adj2 = load_graph(opt.args.name, opt.args.k, 2)
    data = load_data(opt.args.name)
    data1 = torch.Tensor(data.x1).to(opt.args.device)
    data2 = torch.Tensor(data.x2).to(opt.args.device)
    y = data.y


    # train--------------------------
    train(model, data1, data2, adj1, adj2, y)
    print("ACC: {:.4f},".format(opt.args.acc), "NMI: {:.4f},".format(opt.args.nmi), "ARI: {:.4f},".format(opt.args.ari),
          "F1: {:.4f}".format(opt.args.f1))


