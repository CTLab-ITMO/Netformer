
import torch
import torch.nn as nn
from lib.netformer.converter import Converter
from torch.nn import functional as F

SEQ_SIZE = 350
BATCHSIZE = 20

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    with torch.no_grad():
        for data in valid_loader:
            x, y = data
            x, y = x.to(device), y.to(device)
            output = model(x).view(-1)
            mse += ((output - y) ** 2).sum()
    return mse / len(valid_loader)


@torch.enable_grad()
def evaluate_sensitivity(model, valid_dataloader):
    sensitivity = torch.zeros((1, SEQ_SIZE))
    for data, target in valid_dataloader:
        model.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output.flatten(), target)
        loss.backward()
        converter = Converter()
        model_weights = converter.convert(model.layers)[:, 2]
        sensitivity += torch.cat([model_weights.unsqueeze(0), torch.full((1, SEQ_SIZE - model_weights.shape[0]), 0.)],
                                 dim=1)

    sensitivity /= len(valid_dataloader)
    return sensitivity


def get_diff_sense(orig_model, gen_model, valid_dataloader):
    sensitivity_o = evaluate_sensitivity(orig_model, valid_dataloader)
    sensitivity_g = evaluate_sensitivity(gen_model, valid_dataloader)
    return (sensitivity_o - sensitivity_g).abs()



def mixed_loss(y_true, y_pred, sense, edge_c):  # [номер родителя, номер дочки, вес, тип]
    sense = sense.to(device)
    #     y_true, y_pred = y_true.to(device), y_pred.to(device)

    y_true_selected = y_true[:, :, [0, 1, 3]]
    y_pred_selected = y_pred[:, :, [0, 1, 3]]

    loss_vert = nn.MSELoss()(y_true_selected, y_pred_selected)
    loss_edg = (((y_true[:, :, 2] - y_pred[:, :, 2]) ** 2) * sense).sum() * edge_c

    total_loss = loss_edg + loss_vert
    # total_loss = nn.MSELoss(y_true, y_pred)
    return total_loss, loss_vert, loss_edg
