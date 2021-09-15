import tensorflow as tf
from preprocess import load_segments
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
from scipy.special import seterr


USE_LOGISTIC_REGRESSION = True
LEARNING_RATE = 0.0005


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


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
        self.label_mask = tf.placeholder(tf.bool, name='label_mask')
        self.local_adj = self.full_adj[self.local_range, :]
        self.global_deg = np.array(np.sum(self.full_adj, axis=0)).flatten()
        self.global_deg = self.global_deg ** (-1 / 2) # store D^(-1/2)
        self.local_deg = np.array(np.sum(self.local_adj, axis=0)).flatten()
        self.local_deg = self.local_deg ** (-1 / 2)  # store D^(-1/2)
        self.prop_factor = self.full_adj * sp.diags(self.global_deg)  # store AD^(-1/2) for efficient computation
        self.local_prop_factor = sp.diags(self.local_deg) * self.local_adj * sp.diags(self.local_deg)
        with tf.name_scope(self.this_name):
            self.nn_input = tf.placeholder(tf.float32, shape=[None, x_input.shape[-1]], name='nn_input')
            self.w = tf.Variable(tf.zeros([self.nn_input.shape[-1], self.class_num], dtype=tf.float32), trainable=True, name='weight')
            self.b = tf.Variable(tf.zeros([self.class_num], dtype=tf.float32), trainable=False, name='bias')
            self.output = tf.matmul(self.nn_input, self.w, name='output')+self.b
        self.loss = masked_softmax_cross_entropy(self.output, self.label, self.label_mask)
        self.regularize_const = 0.5
        self.loss += self.regularize_const * (tf.nn.l2_loss(self.w) / x_input.shape[-1] + tf.nn.l2_loss(self.b))
        self.learning_rate = 0.01
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.opt_step = self.optimizer.minimize(self.loss)
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.gradients = self.optimizer.compute_gradients(self.loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.this_name))
        self.metrics = masked_accuracy(self.output, self.label, self.label_mask)
        self.new_params = None
        self.lr_model = SGDClassifier(loss='log', fit_intercept=True, learning_rate='adaptive', eta0=learning_rate)

    def get_vertex_id(self):
        # send vertex id to server at initialization phase
        return [self.vertex_id, self.local_range]

    def get_deg_mat(self):
        return self.global_deg

    def get_prop_feature(self):
        prop_result = self.prop_factor * self.feature_layers[-1]  # matmul of sparse matrix
        return prop_result

    def local_prop(self):
        local_result = self.local_prop_factor * self.feature_layers[-1]
        self.feature_layers.append(local_result)

    def set_features(self, new_features):
        self.feature_layers.append(new_features)

    def gen_dense_feature(self):
        self.dense_feature = self.feature_layers[-1].toarray()

    def get_gradient(self):
        return [gv[0] for gv in self.gradients]

    def local_train(self):
        return self.opt_step

    def get_param(self):
        return tf.trainable_variables(scope=self.this_name)

    def set_param(self, params):
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.this_name)
        self.new_params = params
        op = tf.group([tf.assign(var, val) for var, val in zip(var_list, params)])
        return op

    def pred(self):
        return self.metrics

    def get_loss(self):
        return self.loss

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
        loss = - np.sum(prob*np.log(prob)) / samples_idx.shape[0]
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
        with tf.name_scope(self.this_name):
            self.w = tf.Variable(tf.random_normal([self.feature_dim, self.class_num], dtype=tf.float32), trainable=True, name='weight')
            self.b = tf.Variable(tf.zeros([self.class_num], dtype=tf.float32), trainable=True, name='bias')
        self.agg_params = None
        self.learning_rate = 1
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
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

    def collect_grads(self, client_grads: dict):
        agg_grad = []
        for _, grad in client_grads.items():
            for var_grad in grad:
                agg_grad.append(np.zeros_like(var_grad, dtype=np.float32))
            break
        for _, grad in client_grads.items():
            for i in range(len(agg_grad)):
                agg_grad[i] += grad[i]
        for grad in agg_grad:
            grad /= len(client_grads)
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.this_name)
        g_and_v = [(g, v) for g, v in zip(agg_grad, var_list)]
        op = self.optimizer.apply_gradients(g_and_v)
        return op

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
        if from_tf_vars:
            return tf.trainable_variables(scope=self.this_name)
        else:
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
            client = SGCClient(features[cn], adj[cn], clients, vertex_id[cn], cn, labels[cn], learning_rate, min_neighbor)
            self.clients.append(client)
        self.label_masks = label_masks  # {client1:[0, 1, 2], ...}, 0-train mask, 1-val mask, 2-test mask
        self.validate = False
        if len(self.label_masks) == 3:
            self.validate = True
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def init_sys(self):
        ids = {}
        degs = {}
        for c in self.clients:
            ids[c.this_name] = c.get_vertex_id()
            degs[c.this_name] = c.get_deg_mat()
        self.server.collect_ids(ids)
        self.server.collect_degs(degs)

    def propagate(self):
        features = {}
        for c in self.clients:
            features[c.this_name] = c.get_prop_feature()
        self.server.collect_features(features)
        agg = self.server.get_features()
        for c in self.clients:
            c.set_features(agg[c.this_name])

    def local_prop(self):
        for c in self.clients:
            c.local_prop()

    def download_params(self):
        params = self.sess.run(self.server.get_params())
        for c in self.clients:
            self.sess.run(c.set_param(params))

    def train_step(self):
        grads = {}
        total_train_num = 0
        train_acc = 0.0
        train_loss = 0.0
        for c in self.clients:
            train_num = np.where(self.label_masks[c.this_name][0] == True)[0].shape[0]
            total_train_num += train_num
            grad_c, acc, loss = self.sess.run([c.get_gradient(), c.pred(), c.get_loss()], feed_dict={c.nn_input: c.dense_feature, c.label_mask: self.label_masks[c.this_name][0]})
            grads[c.this_name] = grad_c
            train_acc += train_num * acc
            train_loss += train_num * loss
        train_acc /= total_train_num
        train_loss /= total_train_num
        self.sess.run(self.server.collect_grads(grads))
        opt_param = self.sess.run(self.server.get_params())
        for c in self.clients:
            self.sess.run(c.set_param(opt_param))
        return train_acc, train_loss

    def train_step_param(self):
        params = {}
        total_train_num = 0
        train_acc = 0.0
        train_loss = 0.0
        for c in self.clients:
            train_num = np.where(self.label_masks[c.this_name][0] == True)[0].shape[0]
            total_train_num += train_num
            self.sess.run(c.local_train(), feed_dict={c.nn_input: c.dense_feature, c.label_mask: self.label_masks[c.this_name][0]})
            c_params, acc, loss = self.sess.run([c.get_param(), c.pred(), c.get_loss()], feed_dict={c.nn_input: c.dense_feature, c.label_mask: self.label_masks[c.this_name][0]})
            params[c.this_name] = c_params
            train_acc += train_num * acc
            train_loss += train_num * loss
        train_acc /= total_train_num
        train_loss /= total_train_num
        self.server.collect_params(params)
        opt_param = self.server.get_params(from_tf_vars=False)
        for c in self.clients:
            self.sess.run(c.set_param(opt_param))
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
            acc, loss = self.sess.run([c.pred(), c.get_loss()], feed_dict={c.nn_input: c.dense_feature, c.label_mask: self.label_masks[c.this_name][1]})
            val_acc += val_num * acc
            val_loss += val_num * loss
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
            acc, loss = self.sess.run([c.pred(), c.get_loss()], feed_dict={c.nn_input: c.dense_feature, c.label_mask: self.label_masks[c.this_name][-1]})
            test_acc += test_num * acc
            test_loss += test_num * loss
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
        for _ in range(prop_hops):
            if global_prop:
                self.propagate()
            else:
                self.local_prop()
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
            self.download_params()
            for i in range(train_epochs):
                tr_acc, tr_loss = self.train_step()
                # tr_acc, tr_loss = self.train_step_param()
                if verbose:
                    if self.validate:
                        val_acc, val_loss = self.val()
                        print("[Epoch %d] Train Acc: %g, Train Loss: %g, Val Acc: %g, Val Loss: %g" % (i+1, tr_acc, tr_loss, val_acc, val_loss))
                    else:
                        print("[Epoch %d] Train Acc: %g, Train Loss: %g" % (i + 1, tr_acc, tr_loss))
            test_acc, test_loss = self.test()
            res_list.append(test_acc)
            if verbose:
                print("[TEST] TestAcc: %g, TestLoss: %g" % (test_acc, test_loss))
        return res_list


if __name__ == '__main__':
    seterr(all='ignore')
    np.seterr(divide='ignore', invalid='ignore')
    for dataset_name in ['cora', 'pubmed', 'citeseer']:
    # for dataset_name in ['syn-16k']:
        # for dataset_class in ['rnd0', 'rnd1', 'rnd2', 'rnd3']:
        for dataset_class in ['rnd0']:
            clients, vids, adjs, fs, labels, masks = load_segments(dataset_name, 5, dataset_class)
            print("==========%s==========" % dataset_class)
            for flag_global in [False, True]:
                if flag_global:
                    str_global = 'global'
                else:
                    str_global = 'local'
                for prop_step in [2]:
                    if dataset_name == "pubmed":
                        lr_list = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
                    elif dataset_name == "cora":
                        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
                        # lr_list = [0.5]
                    elif dataset_name == "citeseer":
                        lr_list = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
                    else:
                        lr_list = [0.001, 0.005, 0.01, 0.05, 0.1]
                    for lr in lr_list:
                        np.random.seed(0)
                        global_model = FedSGC(clients, adjs, vids, fs, labels, masks, lr, min_neighbor=0)
                        acc = global_model.run(prop_hops=prop_step, train_epochs=200, global_prop=flag_global, verbose=False)
