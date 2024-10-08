import numpy as np
from sklearn.metrics import f1_score, accuracy_score, adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from munkres import Munkres


def cluster_metrics(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "shapes of y_true and y_pred do not match."
    assert len(np.unique(y_true)) == len(np.unique(y_pred)), (f"expected number of clusters {len(np.unique(y_true))}, "
                                                              f"got {len(np.unique(y_pred))}")

    class1 = np.unique(y_true)
    class2 = np.unique(y_pred)

    num_class = len(class1)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cost = np.zeros((num_class, num_class), dtype=int)
    for i, c1 in enumerate(class1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(class2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]

            cost[i][j] = len(mps_d)

    m = Munkres()
    cost = cost.__neg__().tolist()

    indexes = m.compute(cost)

    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(class1):
        c2 = class2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    accuracy = accuracy_score(y_true, new_predict)
    f1 = f1_score(y_true, new_predict, average='macro')
    nmi = normalized_mutual_info_score(y_true, new_predict)
    ari = adjusted_rand_score(y_true, new_predict)
    return accuracy, f1, nmi, ari
