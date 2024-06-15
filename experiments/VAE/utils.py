import torch
import torch.nn as nn
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
