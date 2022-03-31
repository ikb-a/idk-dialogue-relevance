import torch
import torch.nn as nn
import torch.utils.data

class LogisticRegression(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        # Linear layer to combine features
        self.fc1 = nn.Linear(num_features,1)

    def forward(self, x):
        x = self.fc1(x)
        return torch.sigmoid(x)

# For sake of sanity; logistic regression will predict 0 for good matches and 1 for bad matches
def triplet_loss(pos_preds, neg_preds, margin, device):
    """

    :param pos_preds: B -- Batch size
    :param neg_preds: B
    :param margin: in [0,1]
    :return:
    """
    diff = pos_preds - neg_preds + margin
    # Perform max with zero
    losses = torch.where(diff < 0, torch.Tensor([0]).to(device), diff)
    return torch.mean(losses)

# Try to fix gradient decay problems by taking log of distance from max possible triplet loss
def triplet_loss2(pos_preds, neg_preds, margin, device):
    """

    :param pos_preds: B -- Batch size
    :param neg_preds: B
    :param margin: in [0,1]
    :return:
    """
    diff = pos_preds - neg_preds + margin
    # Perform max with zero
    losses = torch.where(diff < 0, torch.Tensor([0]).to(device), diff)
    losses = 1 + margin - losses
    losses = -1 * torch.log(losses)
    return torch.mean(losses)

