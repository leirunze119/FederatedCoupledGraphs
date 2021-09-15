import numpy as np
import sys
import pickle as pkl
import scipy.sparse as sp
import networkx as nx
from sklearn.cluster import KMeans, SpectralClustering
from scipy.spatial.distance import cdist, pdist


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def segment_data(dataset_str, parts):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset_str)
    vertex_num = adj.shape[0]
    train_begin = 0
    val_begin = np.where(val_mask == True)[0][0]
    nolab_begin = np.where(val_mask == True)[0][-1] + 1
    test_begin = np.where(test_mask == True)[0][0]
    train_num = val_begin - train_begin
    val_num = nolab_begin - val_begin
    nolab_num = test_begin - nolab_begin
    test_num = vertex_num - test_begin
    train_per_client = int(np.floor(train_num / parts))
    val_per_client = int(np.floor(val_num / parts))
    nolab_per_client = int(np.floor(nolab_num / parts))
    test_per_client = int(np.floor(test_num / parts))
    train_cnt = 0
    val_cnt = 0
    nolab_cnt = 0
    test_cnt = 0
    clients = []
    adjs = {}
    fs = {}
    vids = {}
    labels = {}
    masks = {}
    label_all = y_train + y_val + y_test
    adj += sp.identity(vertex_num, format='csr')
    for p in range(parts):
        name = str(p)
        clients.append(name)
        vid_local = {}
        if p < parts - 1:
            vid_local['train'] = np.arange(train_begin + train_cnt, train_begin + train_cnt + train_per_client)
            train_cnt += train_per_client
            vid_local['val'] = np.arange(val_begin + val_cnt, val_begin + val_cnt + val_per_client)
            val_cnt += val_per_client
            vid_local['nolab'] = np.arange(nolab_begin + nolab_cnt, nolab_begin + nolab_cnt + nolab_per_client)
            nolab_cnt += nolab_per_client
            vid_local['test'] = np.arange(test_begin + test_cnt, test_begin + test_cnt + test_per_client)
            test_cnt += test_per_client
        else:
            vid_local['train'] = np.arange(train_begin + train_cnt, val_begin)
            vid_local['val'] = np.arange(val_begin + val_cnt, nolab_begin)
            vid_local['nolab'] = np.arange(nolab_begin + nolab_cnt, test_begin)
            vid_local['test'] = np.arange(test_begin + test_cnt, vertex_num)
        vid_local_list = np.concatenate([vid_local['train'], vid_local['val'], vid_local['nolab'], vid_local['test']])
        adj_local = adj[:, vid_local_list]
        deg_local = np.array(np.sum(adj_local, axis=-1)).flatten()
        vid_adj = np.where(deg_local > 0)[0]
        cols_in_adj = []
        for i in vid_local_list:
            cols_in_adj.append(np.where(vid_adj == i)[0][0])
        vids[name] = [vid_adj, cols_in_adj]
        adjs[name] = adj_local[vid_adj, :]
        fs[name] = features[vid_local_list, :]
        labels[name] = label_all[vid_local_list, :]
        train_mask_c = np.array([False] * vertex_num)
        train_mask_c[vid_local['train']] = True
        train_mask_c = train_mask_c[vid_local_list]
        val_mask_c = np.array([False] * vertex_num)
        val_mask_c[vid_local['val']] = True
        val_mask_c = val_mask_c[vid_local_list]
        val_labels = label_all[np.where(val_mask_c == True)[0], :]
        test_mask_c = np.array([False] * vertex_num)
        test_mask_c[vid_local['test']] = True
        test_mask_c = test_mask_c[vid_local_list]
        masks[name] = [train_mask_c, val_mask_c, test_mask_c]
    # vids_by_clients = {}
    # for c_local in clients:
    #     seg_vids = {}
    #     for c_peer in clients:
    #         uni_set = set(vids_adj[c_local])
    #         peer_set = set(vids_local[c_peer])
    #         inter = uni_set & peer_set
    #         seg_vids[c_peer] =
    with open("%s_%d.pkl" % (dataset_str, parts), 'wb') as f:
        pkl.dump([clients, vids, adjs, fs, labels, masks], f, protocol=pkl.HIGHEST_PROTOCOL)


def random_segment(dataset_str, parts, seed=0):
    rnd_seed = seed
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset_str)
    vertex_num = adj.shape[0]
    train_begin = 0
    val_begin = np.where(val_mask == True)[0][0]
    nolab_begin = np.where(val_mask == True)[0][-1] + 1
    test_begin = np.where(test_mask == True)[0][0]
    train_perm = np.random.permutation(np.arange(train_begin, val_begin))
    val_perm = np.random.permutation(np.arange(val_begin, nolab_begin))
    nolab_perm = np.random.permutation(np.arange(nolab_begin, test_begin))
    test_perm = np.random.permutation(np.arange(test_begin, vertex_num))
    full_perm = np.concatenate([train_perm, val_perm, nolab_perm, test_perm])
    adj = adj[full_perm, :]
    adj = adj[:, full_perm]
    features = features[full_perm, :]
    y_train = y_train[full_perm, :]
    y_val = y_val[full_perm, :]
    y_test = y_test[full_perm, :]
    train_mask = train_mask[full_perm]
    val_mask = val_mask[full_perm]
    test_mask = test_mask[full_perm]
    train_num = val_begin - train_begin
    val_num = nolab_begin - val_begin
    nolab_num = test_begin - nolab_begin
    test_num = vertex_num - test_begin
    train_per_client = int(np.floor(train_num / parts))
    val_per_client = int(np.floor(val_num / parts))
    nolab_per_client = int(np.floor(nolab_num / parts))
    test_per_client = int(np.floor(test_num / parts))
    train_cnt = 0
    val_cnt = 0
    nolab_cnt = 0
    test_cnt = 0
    clients = []
    adjs = {}
    fs = {}
    vids = {}
    labels = {}
    masks = {}
    label_all = y_train + y_val + y_test
    adj += sp.identity(vertex_num, format='csr')
    for p in range(parts):
        name = str(p)
        clients.append(name)
        vid_local = {}
        if p < parts - 1:
            vid_local['train'] = np.arange(train_begin + train_cnt, train_begin + train_cnt + train_per_client)
            train_cnt += train_per_client
            vid_local['val'] = np.arange(val_begin + val_cnt, val_begin + val_cnt + val_per_client)
            val_cnt += val_per_client
            vid_local['nolab'] = np.arange(nolab_begin + nolab_cnt, nolab_begin + nolab_cnt + nolab_per_client)
            nolab_cnt += nolab_per_client
            vid_local['test'] = np.arange(test_begin + test_cnt, test_begin + test_cnt + test_per_client)
            test_cnt += test_per_client
        else:
            vid_local['train'] = np.arange(train_begin + train_cnt, val_begin)
            vid_local['val'] = np.arange(val_begin + val_cnt, nolab_begin)
            vid_local['nolab'] = np.arange(nolab_begin + nolab_cnt, test_begin)
            vid_local['test'] = np.arange(test_begin + test_cnt, vertex_num)
        vid_local_list = np.concatenate([vid_local['train'], vid_local['val'], vid_local['nolab'], vid_local['test']])
        adj_local = adj[:, vid_local_list]
        deg_local = np.array(np.sum(adj_local, axis=-1)).flatten()
        vid_adj = np.where(deg_local > 0)[0]
        cols_in_adj = []
        for i in vid_local_list:
            cols_in_adj.append(np.where(vid_adj == i)[0][0])
        vids[name] = [vid_adj, cols_in_adj]
        adjs[name] = adj_local[vid_adj, :]
        fs[name] = features[vid_local_list, :]
        labels[name] = label_all[vid_local_list, :]
        train_mask_c = np.array([False] * vertex_num)
        train_mask_c[vid_local['train']] = True
        train_mask_c = train_mask_c[vid_local_list]
        val_mask_c = np.array([False] * vertex_num)
        val_mask_c[vid_local['val']] = True
        val_mask_c = val_mask_c[vid_local_list]
        val_labels = label_all[np.where(val_mask_c == True)[0], :]
        test_mask_c = np.array([False] * vertex_num)
        test_mask_c[vid_local['test']] = True
        test_mask_c = test_mask_c[vid_local_list]
        masks[name] = [train_mask_c, val_mask_c, test_mask_c]
    with open("%s_rnd%d_%d.pkl" % (dataset_str, rnd_seed, parts), 'wb') as f:
        pkl.dump([clients, vids, adjs, fs, labels, masks], f, protocol=pkl.HIGHEST_PROTOCOL)


def kmeans_segment(dataset_str, parts):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset_str)
    vertex_num = adj.shape[0]
    train_begin = 0
    label_mask = np.logical_or(np.logical_or(train_mask, val_mask), test_mask)
    label_idx = np.where(label_mask == True)[0]
    label_num = label_idx.shape[0]
    train_num = np.where(train_mask == True)[0].shape[0]
    train_idx = label_idx[:train_num]
    test_idx = label_idx[train_num:]
    nolab_idx = np.where(label_mask == False)[0]
    cluster_alg = KMeans(parts).fit(features)
    cluster_res = cluster_alg.predict(features)
    clients = []
    adjs = {}
    fs = {}
    vids = {}
    labels = {}
    masks = {}
    label_all = y_train + y_val + y_test
    adj += sp.identity(vertex_num, format='csr')
    for p in range(parts):
        name = str(p)
        clients.append(name)
        vid_local_list = np.where(cluster_res == p)[0]
        vid_local = {'train': [], 'test': []}
        for vid in vid_local_list:
            if vid in train_idx:
                vid_local['train'].append(vid)
            elif vid in test_idx:
                vid_local['test'].append(vid)
        adj_local = adj[:, vid_local_list]
        deg_local = np.array(np.sum(adj_local, axis=-1)).flatten()
        vid_adj = np.where(deg_local > 0)[0]
        cols_in_adj = []
        for i in vid_local_list:
            cols_in_adj.append(np.where(vid_adj == i)[0][0])
        vids[name] = [vid_adj, cols_in_adj]
        adjs[name] = adj_local[vid_adj, :]
        fs[name] = features[vid_local_list, :]
        labels[name] = label_all[vid_local_list, :]
        train_mask_c = np.array([False] * vertex_num)
        train_mask_c[vid_local['train']] = True
        train_mask_c = train_mask_c[vid_local_list]
        test_mask_c = np.array([False] * vertex_num)
        test_mask_c[vid_local['test']] = True
        test_mask_c = test_mask_c[vid_local_list]
        masks[name] = [train_mask_c, test_mask_c]
    with open("%s_kmeans20_%d.pkl" % (dataset_str, parts), 'wb') as f:
        pkl.dump([clients, vids, adjs, fs, labels, masks], f, protocol=pkl.HIGHEST_PROTOCOL)


def connection_segment(dataset_str, parts):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset_str)
    vertex_num = adj.shape[0]
    train_begin = 0
    val_begin = np.where(val_mask == True)[0][0]
    nolab_begin = np.where(val_mask == True)[0][-1] + 1
    test_begin = np.where(test_mask == True)[0][0]
    train_num = val_begin - train_begin
    val_num = nolab_begin - val_begin
    nolab_num = test_begin - nolab_begin
    test_num = vertex_num - test_begin
    train_per_client = int(np.floor(train_num / parts))
    val_per_client = int(np.floor(val_num / parts))
    nolab_per_client = int(np.floor(nolab_num / parts))
    test_per_client = int(np.floor(test_num / parts))
    train_cnt = 0
    val_cnt = 0
    nolab_cnt = 0
    test_cnt = 0
    clients = []
    adjs = {}
    fs = {}
    vids = {}
    labels = {}
    masks = {}
    label_all = y_train + y_val + y_test
    adj += sp.identity(vertex_num, format='csr')
    train_budgets = np.ones(5, dtype=np.int32) * train_per_client
    train_diff = train_num - np.sum(train_budgets)
    for i in range(train_diff):
        train_budgets[i] += 1
    val_budgets = np.ones(5, dtype=np.int32) * val_per_client
    val_diff = val_num - np.sum(val_budgets)
    for i in range(val_diff):
        val_budgets[i] += 1
    nolab_budgets = np.ones(5, dtype=np.int32) * nolab_per_client
    nolab_diff = nolab_num - np.sum(nolab_budgets)
    for i in range(nolab_diff):
        nolab_budgets[i] += 1
    test_budgets = np.ones(5, dtype=np.int32) * test_per_client
    test_diff = test_num - np.sum(test_budgets)
    for i in range(test_diff):
        test_budgets[i] += 1
    vid_locals = {}
    for p in range(parts):
        name = str(p)
        clients.append(name)
        vid_local = {}
        if p < parts - 1:
            vid_local['train'] = np.arange(train_begin + train_cnt, train_begin + train_cnt + train_per_client)
            train_cnt += train_per_client
            vid_local['val'] = np.arange(val_begin + val_cnt, val_begin + val_cnt + val_per_client)
            val_cnt += val_per_client
            vid_local['nolab'] = np.arange(nolab_begin + nolab_cnt, nolab_begin + nolab_cnt + nolab_per_client)
            nolab_cnt += nolab_per_client
            vid_local['test'] = np.arange(test_begin + test_cnt, test_begin + test_cnt + test_per_client)
            test_cnt += test_per_client
        else:
            vid_local['train'] = np.arange(train_begin + train_cnt, val_begin)
            vid_local['val'] = np.arange(val_begin + val_cnt, nolab_begin)
            vid_local['nolab'] = np.arange(nolab_begin + nolab_cnt, test_begin)
            vid_local['test'] = np.arange(test_begin + test_cnt, vertex_num)
        vid_local_list = np.concatenate([vid_local['train'], vid_local['val'], vid_local['nolab'], vid_local['test']])
        adj_local = adj[:, vid_local_list]
        deg_local = np.array(np.sum(adj_local, axis=-1)).flatten()
        vid_adj = np.where(deg_local > 0)[0]
        cols_in_adj = []
        for i in vid_local_list:
            cols_in_adj.append(np.where(vid_adj == i)[0][0])
        vids[name] = [vid_adj, cols_in_adj]
        adjs[name] = adj_local[vid_adj, :]
        fs[name] = features[vid_local_list, :]
        labels[name] = label_all[vid_local_list, :]
        train_mask_c = np.array([False] * vertex_num)
        train_mask_c[vid_local['train']] = True
        train_mask_c = train_mask_c[vid_local_list]
        val_mask_c = np.array([False] * vertex_num)
        val_mask_c[vid_local['val']] = True
        val_mask_c = val_mask_c[vid_local_list]
        val_labels = label_all[np.where(val_mask_c == True)[0], :]
        test_mask_c = np.array([False] * vertex_num)
        test_mask_c[vid_local['test']] = True
        test_mask_c = test_mask_c[vid_local_list]
        masks[name] = [train_mask_c, val_mask_c, test_mask_c]
    with open("%s_minadj_%d.pkl" % (dataset_str, parts), 'wb') as f:
        pkl.dump([clients, vids, adjs, fs, labels, masks], f, protocol=pkl.HIGHEST_PROTOCOL)


def load_segments(dataset_str, segments, preproc=None, nn=None):
    if preproc is None:
        d_name = "{}_{}".format(dataset_str, segments)
    else:
        d_name = "{}_{}_{}".format(dataset_str, preproc, segments)
    if nn is None:
        pass
    else:
        d_name += "-nn{}".format(nn)
    d_name += ".pkl"
    with open(d_name, 'rb') as f:
        clients, vids, adjs, fs, labels, masks = pkl.load(f)
    return clients, vids, adjs, fs, labels, masks


def add_local_connections(dataset_str, segments, preproc=None, min_neighbor=3):
    clients, vids, adjs, fs, labels, masks = load_segments(dataset_str, segments, preproc)
    for c in clients:
        local_range = np.array(vids[c][1], dtype=np.int32)
        full_adj = adjs[c]
        feature = fs[c].toarray()
        for v_ref_col, v_ref in enumerate(local_range):
            need_neighbor = min_neighbor + 1 - np.sum(full_adj[v_ref, :])
            if need_neighbor > 0:
                distances = cdist(feature[v_ref_col, :].reshape([1, -1]), feature, 'euclidean').flatten()
                neighbors = np.argsort(distances)
                connections = 0
                for v_col, v in enumerate(local_range[neighbors]):
                    if full_adj[v_ref, v_col] == 0:
                        connections += 1
                        full_adj[v_ref, v_col] = 1
                        full_adj[v, v_ref_col] = 1
                    if connections >= need_neighbor:
                        break
        adjs[c] = full_adj
    with open("%s_%s_neighbor%d_%d.pkl" % (dataset_str, preproc, min_neighbor, segments), 'wb') as f:
        pkl.dump([clients, vids, adjs, fs, labels, masks], f, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    for d_name in ['cora', 'citeseer', 'pubmed']:
        for seg in [1, 2, 3, 4, 5]:
            random_segment(d_name, seg, seed=4)
    # for d_name in ['citeseer', 'cora', 'pubmed']:
    #     add_local_connections(d_name, 5, 'best', min_neighbor=1)


