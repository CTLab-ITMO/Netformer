import torch
import torch.nn.functional as F
from pymfe.mfe import MFE
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def edge_metric(net_matrix_true, net_matrix_pred):
    false_edges = (((net_matrix_true != 0) == (net_matrix_pred != 0)) == False).sum() / (net_matrix_true != 0).sum()
    return false_edges

def valid_cos_sim(model1, model2, valid_loader):
    cos_sim = 0
    with torch.no_grad():
        for data in valid_loader:
            x, y = data
            x, y = x.to(device), y.to(device)
            model1(x)
            model2(x)

            cos_sim += (F.cosine_similarity(model1.emb, model2.emb).sum() / x.shape[0])
    return cos_sim / len(valid_loader)


def valid_model(model, valid_loader):
    mse = 0
    cos_sim = 0
    with torch.no_grad():
        for data in valid_loader:
            x, y = data

            x, y = x.to(device), y.to(device)
            output = model(x).view(-1)
            mse += ((output - y) ** 2).sum()
    return mse / len(valid_loader)

def get_meta_features(data):
    s = "inst_to_attr, nr_class, nr_attr, attr_to_inst, skewness, kurtosis, cor, cov, attr_conc, class_conc, sparsity, gravity, skewness, class_ent, attr_ent, mut_inf, eq_num_attr, ns_ratio, f1, f2, tree_depth, leaves_branch, nodes_per_attr, leaves_per_class"
    s = s.split(", ")

    X, y = data.drop(['reg_id', 'y'], axis=1).to_numpy(), data['y'].to_numpy()

    mfe = MFE(features=[*s])
    mfe.fit(X, y)
    ft = mfe.extract()
    return ft[1]
