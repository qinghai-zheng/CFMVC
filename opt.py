import argparse

parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# setting
'''
   BBCSport
'''

dataset = 'BBCSport'
parser.add_argument('--name', type=str, default=dataset)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--sigma', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_clusters', default=7, type=int)
parser.add_argument('--n_z', type=int, default=10)
parser.add_argument('--pretrain_path1', type=str, default='pkl')
parser.add_argument('--pretrain_path2', type=str, default='pkl')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=3)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--acc', type=float, default=-1)
parser.add_argument('--p1', type=float, default=10)
parser.add_argument('--p2', type=float, default=10)

args = parser.parse_args()
