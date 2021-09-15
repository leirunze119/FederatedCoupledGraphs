from preprocess import load_segments
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
from scipy.special import seterr
from os.path import isfile
import pickle as pkl

USE_SAVED_PROP = False
USE_LOGISTIC_REGRESSION = True
LEARNING_RATE = 0.0005

g_task_name = ""


class SGCClient:
    def __init__(self,
                 x_input,
                 adj,
                 peers: list,
                 vertex_id: list,
                 this_name: str,
                 labels,
                 learning_rate=LEARNING_RATE,
                 min_neighbor=0):
        # x_input: tf-variable or placeholder, #row = #vertex in subgraph, #col = #feature
        # adj: preprocessed adjacency matrix, #col = #vertex in subgraph, #row = #vertex adjacent to this subgraph
        #      vertices from the same subgraph should be arranged in continuous rows
        #      identical matrix should be added to the ``local'' area before input
        # peers: a list object, each element is the name of a adjacent subgraph (include itself), same order as adj
        # vertex_id: a list object, 0 = id of all related vertexes
        #                           1 = range of local vertexes in col
        # my_name: a str of name of this user
        self.feature_layers = [x_input]
        self.dense_feature = None
        self.full_adj = adj
        self.peers = peers
        self.this_name = this_name
        self.vertex_id = np.array(vertex_id[0], dtype=np.int32)
        self.local_range = np.array(vertex_id[1], dtype=np.int32)
        self.min_local_neighbor = min_neighbor
        if self.min_local_neighbor > 0:
            for v_ref_col, v_ref in enumerate(self.local_range):
                need_neighbor = self.min_local_neighbor + 1 - np.sum(self.full_adj[v_ref, :])
                if need_neighbor > 0:
                    distances = []
                    for v_col, v in enumerate(self.local_range):
                        distances.append(np.linalg.norm((x_input[v_ref_col, :] - x_input[v_col, :]).toarray()))
                    neighbors = np.argsort(distances)
                    connections = 0
                    for v_col, v in enumerate(self.local_range[neighbors]):
                        if self.full_adj[v_ref, v_col] == 0:
                            connections += 1
                            self.full_adj[v_ref, v_col] = 1
                            self.full_adj[v, v_ref_col] = 1
                        if connections >= need_neighbor:
                            break
        self.label = labels
        self.class_num = self.label.shape[-1]
        self.label_int = np.argmax(self.label, -1)
        self.classes = np.arange(self.class_num)
        self.local_adj = self.full_adj[self.local_range, :]
        self.global_deg = np.array(np.sum(self.full_adj, axis=0)).flatten()
        self.global_deg = self.global_deg ** (-1 / 2)  # store D^(-1/2)
        self.local_deg = np.array(np.sum(self.local_adj, axis=0)).flatten()
        self.local_deg = self.local_deg ** (-1 / 2)  # store D^(-1/2)
        self.prop_factor = self.full_adj * sp.diags(self.global_deg)  # store AD^(-1/2) for efficient computation
        self.local_prop_factor = sp.diags(self.local_deg) * self.local_adj * sp.diags(self.local_deg)
        self.regularize_const = 0.5
        self.learning_rate = 0.01
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.new_params = None
        self.lr_model = SGDClassifier(loss='log', fit_intercept=True, learning_rate='adaptive', eta0=learning_rate)

    def get_vertex_id(self):
        # send vertex id to server at initialization phase
        return [self.vertex_id, self.local_range]

    def get_deg_mat(self):
        return self.global_deg

    def get_prop_feature(self):
        prop_result = self.prop_factor * self.feature_layers[-1]  # matmul of sparse matrix
        full_cnt = prop_result.data.shape[0]
        local_prop = prop_result[self.local_range, :]
        local_cnt = local_prop.data.shape[0]
        return prop_result

    def local_prop(self):
        local_result = self.local_prop_factor * self.feature_layers[-1]
        self.feature_layers.append(local_result)

    def set_features(self, new_features):
        self.feature_layers.append(new_features)

    def gen_dense_feature(self):
        self.dense_feature = self.feature_layers[-1].toarray()
        # self.dense_feature = self.feature_layers[-1]

    def get_lr_param(self):
        return [self.lr_model.coef_, self.lr_model.intercept_]

    def set_lr_param(self, lr_params):
        self.lr_model.coef_, self.lr_model.intercept_ = lr_params

    def lr_train(self, label_mask):
        samples_idx = np.where(label_mask == True)[0]
        self.lr_model.partial_fit(self.dense_feature[samples_idx, :], self.label_int[samples_idx], classes=self.classes)

    def lr_pred(self, label_mask):
        samples_idx = np.where(label_mask == True)[0]
        acc = self.lr_model.score(self.dense_feature[samples_idx, :], self.label_int[samples_idx])
        prob = self.lr_model.predict_proba(self.dense_feature[samples_idx, :])
        loss = - np.sum(prob * np.log(prob)) / samples_idx.shape[0]
        return acc, loss


class SGCServer:
    def __init__(self, clients: list,
                 this_name: str,
                 class_num,
                 feature_dim):
        self.this_name = this_name
        self.clients = clients
        self.class_num = class_num
        self.feature_dim = feature_dim
        self.global_ids = np.empty(0, dtype=np.int32)
        self.client_range = {}
        self.align = {}
        self.vertex_num = 0
        self.deg_buffer = np.empty(0, dtype=np.float32)
        self.agg_params = None
        self.learning_rate = 1
        self.feature_buffer = None

    def collect_ids(self, ids: dict):
        for client_name in self.clients:
            self.client_range[client_name] = [self.global_ids.shape[0]]
            self.global_ids = np.concatenate([self.global_ids, ids[client_name][0][ids[client_name][1]]])
            self.client_range[client_name].append(self.global_ids.shape[0])
        for client_name in self.clients:
            client_align = []
            for align_id in ids[client_name][0]:
                client_align.append(np.where(self.global_ids == align_id)[0][0])
            self.align[client_name] = np.array(client_align, dtype=np.int32)
        self.vertex_num = len(self.global_ids)

    def collect_degs(self, degs: dict):
        for client_name in self.clients:
            self.deg_buffer = np.concatenate([self.deg_buffer, degs[client_name]])

    def collect_features(self, features: dict):
        self.feature_buffer = sp.csr_matrix((self.vertex_num, self.feature_dim), dtype=np.float32)
        for client_name, local_res in features.items():
            self.feature_buffer[self.align[client_name], :] += local_res
        self.feature_buffer = sp.diags(self.deg_buffer) * self.feature_buffer  # mat mul

    def get_features(self):
        split_features = {}
        for client_name, r in self.client_range.items():
            split_features[client_name] = self.feature_buffer[r[0]:r[1], :]
        return split_features

    def collect_params(self, client_params: dict):
        self.agg_params = []
        for _, param in client_params.items():
            for var_param in param:
                # self.agg_params.append(np.zeros_like(var_param, dtype=np.float32))
                self.agg_params.append(np.zeros_like(var_param))
            break
        for _, param in client_params.items():
            for i in range(len(self.agg_params)):
                self.agg_params[i] += param[i]
        for param in self.agg_params:
            param /= len(client_params)

    def get_params(self, from_tf_vars=True):
            return self.agg_params


class FedSGC:
    def __init__(self, clients: list,
                 adj: dict,
                 vertex_id: dict,
                 features: dict,
                 labels: dict,
                 label_masks: dict,
                 learning_rate=LEARNING_RATE,
                 min_neighbor=0):
        self.class_num = labels[clients[0]].shape[-1]
        self.feature_dim = features[clients[0]].shape[-1]
        self.server = SGCServer(clients, 'server', self.class_num, self.feature_dim)
        self.clients = []
        for cn in clients:
            client = SGCClient(features[cn], adj[cn], clients, vertex_id[cn], cn, labels[cn], learning_rate,
                               min_neighbor)
            self.clients.append(client)
        self.label_masks = label_masks  # {client1:[0, 1, 2], ...}, 0-train mask, 1-val mask, 2-test mask
        self.validate = False
        # if len(self.label_masks) == 3:
        #     self.validate = True

    def init_sys(self):
        ids = {}
        degs = {}
        for c in self.clients:
            ids[c.this_name] = c.get_vertex_id()
            degs[c.this_name] = c.get_deg_mat()
        self.server.collect_ids(ids)
        self.server.collect_degs(degs)

    def propagate(self, iter_id):
        if USE_SAVED_PROP and isfile("{}-{}.prop".format(g_task_name, iter_id+1)):
            with open("{}-{}.prop".format(g_task_name, iter_id+1), "rb") as prop_f:
                agg = pkl.load(prop_f)
                for c in self.clients:
                    c.set_features(agg[c.this_name])
        else:
            features = {}
            cnt_client = []
            for c in self.clients:
                features[c.this_name] = c.get_prop_feature()
            self.server.collect_features(features)
            agg = self.server.get_features()
            for c in self.clients:
                c.set_features(agg[c.this_name])
            if USE_SAVED_PROP:
                with open("{}-{}.prop".format(g_task_name, iter_id + 1), "wb") as prop_f:
                    pkl.dump(agg, prop_f)

    def local_prop(self, iter_id):
        if USE_SAVED_PROP and isfile("{}-{}.prop".format(g_task_name, iter_id + 1)):
            with open("{}-{}.prop".format(g_task_name, iter_id+1), "rb") as prop_f:
                feats = pkl.load(prop_f)
                for c in self.clients:
                    c.set_features(feats[c.this_name])
        else:
            cnt_client = []
            for c in self.clients:
                c.local_prop()
            if USE_SAVED_PROP:
                feats = {}
                for c in self.clients:
                    feats[c.this_name] = c.feature_layers[-1]
                with open("{}-{}.prop".format(g_task_name, iter_id + 1), "wb") as prop_f:
                    pkl.dump(feats, prop_f)

    def train_step(self):
        grads = {}
        total_train_num = 0
        train_acc = 0.0
        train_loss = 0.0
        for c in self.clients:
            train_num = np.where(self.label_masks[c.this_name][0] == True)[0].shape[0]
            total_train_num += train_num
            train_acc += train_num * acc
        train_acc /= total_train_num
        train_loss /= total_train_num
        return train_acc, train_loss

    def train_step_param(self):
        params = {}
        total_train_num = 0
        train_acc = 0.0
        train_loss = 0.0
        for c in self.clients:
            train_num = np.where(self.label_masks[c.this_name][0] == True)[0].shape[0]
            total_train_num += train_num
            train_acc += train_num * acc
        train_acc /= total_train_num
        train_loss /= total_train_num
        self.server.collect_params(params)
        opt_param = self.server.get_params(from_tf_vars=False)
        return train_acc, train_loss

    def train_step_lr(self):
        params = {}
        for c in self.clients:
            c.lr_train(self.label_masks[c.this_name][0])
            params[c.this_name] = c.get_lr_param()
        self.server.collect_params(params)
        opt_param = self.server.get_params(from_tf_vars=False)
        for c in self.clients:
            c.set_lr_param(opt_param)
        return self.test_lr(mode='train')

    def val(self):
        total_val_num = 0
        val_acc = 0.0
        val_loss = 0.0
        for c in self.clients:
            val_num = np.where(self.label_masks[c.this_name][1] == True)[0].shape[0]
            total_val_num += val_num
            val_acc += val_num * acc
        val_acc /= total_val_num
        val_loss /= total_val_num
        return val_acc, val_loss

    def test(self):
        total_test_num = 0
        test_acc = 0.0
        test_loss = 0.0
        for c in self.clients:
            test_num = np.where(self.label_masks[c.this_name][-1] == True)[0].shape[0]
            total_test_num += test_num
            test_acc += test_num * acc
        test_acc /= total_test_num
        test_loss /= total_test_num
        return test_acc, test_loss

    def test_lr(self, mode='test'):
        mask_id = 2
        if mode == 'train':
            mask_id = 0
        elif mode == 'val':
            mask_id = 1
        elif mode == 'test':
            mask_id = -1
        total_test_num = 0
        test_acc = 0.0
        test_loss = 0.0
        for c in self.clients:
            test_num = np.where(self.label_masks[c.this_name][mask_id] == True)[0].shape[0]
            total_test_num += test_num
            acc, loss = c.lr_pred(self.label_masks[c.this_name][mask_id])
            test_acc += test_num * acc
            test_loss += test_num * loss
        test_acc /= total_test_num
        test_loss /= total_test_num
        return test_acc, test_loss

    def run(self, prop_hops, train_epochs, global_prop=True, verbose=True):
        self.init_sys()
        for it in range(prop_hops):
            if global_prop:
                self.propagate(it)
            else:
                self.local_prop(it)
        for c in self.clients:
            c.gen_dense_feature()
        res_list = []
        if USE_LOGISTIC_REGRESSION:
            for i in range(train_epochs):
                tr_acc, tr_loss = self.train_step_lr()
                if verbose:
                    if self.validate:
                        val_acc, val_loss = self.test_lr(mode='val')
                        print("[Epoch %d] Train Acc: %g, Train Loss: %g, Val Acc: %g, Val Loss: %g" % (
                            i + 1, tr_acc, tr_loss, val_acc, val_loss))
                    else:
                        print("[Epoch %d] Train Acc: %g, Train Loss: %g" % (i + 1, tr_acc, tr_loss))
                test_acc, test_loss = self.test_lr(mode='test')
                res_list.append(test_acc)
                if verbose:
                    print("[TEST] TestAcc: %g, TestLoss: %g" % (test_acc, test_loss))
        else:
            for i in range(train_epochs):
                tr_acc, tr_loss = self.train_step()
                # tr_acc, tr_loss = self.train_step_param()
                if verbose:
                    if self.validate:
                        val_acc, val_loss = self.val()
                        print("[Epoch %d] Train Acc: %g, Train Loss: %g, Val Acc: %g, Val Loss: %g" % (
                        i + 1, tr_acc, tr_loss, val_acc, val_loss))
                    else:
                        print("[Epoch %d] Train Acc: %g, Train Loss: %g" % (i + 1, tr_acc, tr_loss))
            test_acc, test_loss = self.test()
            res_list.append(test_acc)
            if verbose:
                print("[TEST] TestAcc: %g, TestLoss: %g" % (test_acc, test_loss))
        return res_list


if __name__ == '__main__':
    # seg_num = 2
    for seg_num in [1, 2, 3, 5, 10]:
        results = []
        # for dataset_name in ['syn-16k']:
        for dataset_name in ['cora', 'pubmed', 'citeseer', 'coauthor_cs', 'coauthor_phy', 'flickr']:
            for dataset_class in ['rnd0', 'rnd1', 'rnd2', 'rnd3', 'rnd4']:
                clients, vids, adjs, fs, labels, masks = load_segments(dataset_name, seg_num, dataset_class)
                print("==========%s==========" % dataset_class)
                if seg_num > 1:
                    global_list = [True, False]
                else:
                    global_list = [False]
                for flag_global in global_list:
                    if flag_global:
                        str_global = 'global'
                    else:
                        str_global = 'local'
                    for prop_step in [2]:
                        if dataset_name == "pubmed":
                            lr_list = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
                        elif dataset_name == "cora":
                            lr_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
                        elif dataset_name == "citeseer":
                            lr_list = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
                        elif dataset_name == "flickr":
                            lr_list = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005]
                        elif dataset_name == "coauthor_phy":
                            lr_list = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005]
                        elif dataset_name == "coauthor_cs":
                            lr_list = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005]
                        else:
                            lr_list = [0.1, 0.05, 0.01, 0.005, 0.001]
                        for lr in lr_list:
                            np.random.seed(0)
                            g_task_name = "{}-seg{}-{}-{}".format(dataset_name, seg_num, dataset_class, str_global)
                            global_model = FedSGC(clients, adjs, vids, fs, labels, masks, lr, min_neighbor=0)
                            acc = global_model.run(prop_hops=prop_step, train_epochs=200, global_prop=flag_global,
                                                       verbose=False)
