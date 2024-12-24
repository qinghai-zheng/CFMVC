import h5py
import scipy.io as scio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize

topk = 10
def construct_graph(dataset, features, label, method='ncos', i=0):
    print('----------------------------{}  view{}  method: {}'.format(dataset, i, method))

    fname = 'graph/{}10{}_graph.txt'.format(dataset, i)
    num = len(label)
    dist = None

    if method == 'heat':
        dist = -0.5 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        #features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        #features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    print(num)
    print(features)
    print(dist)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk+1))[-(topk+1):]
        inds.append(ind)

    #print(inds)
    f = open(fname, 'w')
    counter = 0
    n=0
    A = np.zeros_like(dist)
    for i, v in enumerate(inds):
        mutual_knn = False
        m=0
        for vv in v:
            if vv == i:
                pass
            else:
                if m!=10:
                    if label[vv] != label[i]:
                        counter += 1
                    f.write('{} {}\n'.format(i, vv))
                    n += 1
                    m += 1

    f.close()

    print('error rate: {}'.format(counter / (num * topk)))
    print(n)





dataset='3Sources'



if dataset == 'BBCSport':
    # methods: 'ncos'
    x1 = scio.loadmat('data/{}.mat'.format(dataset))['X1']
    x2 = scio.loadmat('data/{}.mat'.format(dataset))['X2']
    y = scio.loadmat('data/{}.mat'.format(dataset))['gt']
    x1 = np.transpose(x1)
    x2 = np.transpose(x2)
    y = np.squeeze(y)



for i in[1, 2]:
    if i == 1:
        construct_graph(dataset, x1, y, 'ncos', i)
    if i == 2:
        construct_graph(dataset, x2, y, 'ncos', i)




