import opt
import numpy as np
from munkres import Munkres
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics.cluster import adjusted_rand_score as ari_score
from sklearn import metrics
from sklearn.cluster import KMeans


def clustering(Z, y):
    """
    clustering based on embedding
    Args:
        Z: the input embedding
        y: the ground truth

    Returns: acc, nmi, ari, f1, clustering centers
    """
    kmeans = KMeans(n_clusters=opt.args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(Z.data.cpu().numpy())
    acc, nmi, ari, f1 = eva(y, y_pred, 'Z')
    return acc, nmi, ari, f1


def cluster_acc(y_true, y_pred):
    """
        calculate clustering acc and f1-score
        Args:
            y_true: the ground truth
            y_pred: the clustering id

        Returns: acc and f1-score
        """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    numclass1 = len(l1)
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if numclass1 != numclass2:
        print('error')
        return
    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro


def eva(y_true, y_pred, epoch=0):
    """
        evaluate the clustering performance
        Args:
            y_true: the ground truth
            y_pred: the predicted label
            show_details: if print the details
        Returns: None
        """
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    # print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
    #         ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1


