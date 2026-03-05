from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable


class OLELoss(nn.Module):
    def __init__(self, lambda_=0.25, reduction='mean'):
        super(OLELoss, self).__init__()
        self.lambda_ = lambda_
        self.reduction = reduction

    def forward(self, features, labels):
        N, D = features.size()

        labels = labels.unsqueeze(1)
        matched = (labels == labels.t()).float()

        expanded_features = features.unsqueeze(1)
        diffs = expanded_features - expanded_features.transpose(0, 1)
        norms = torch.norm(diffs, dim=2)

        intra_loss = torch.sum(norms * matched)

        not_matched = 1 - matched
        inter_loss = torch.sum((1 / (norms + 1e-6)) * not_matched)

        total_loss = intra_loss + self.lambda_ * inter_loss
        if self.reduction == 'mean':
            total_loss /= N * N

        return total_loss

class OLELoss_uni(nn.Module):
    def __init__(self, lambda_=0.25, reduction='sum'):
        super(OLELoss_uni, self).__init__()
        self.lambda_ = lambda_
        self.reduction = reduction

    def pairwise_sampling(self, img_features, txt_features):
        N = len(img_features)
        matched = self.full_sampling(N)
        matched = torch.tensor(matched, dtype=torch.float, device=img_features.device).view(N, N)
        return matched
    # This is the full sampling function
    def full_sampling(self, N):
        matched = []
        for i in range(N):
            for j in range(N):
                if i == j:
                    matched.append(1)
                else:
                    matched.append(0)
        return matched

    def forward(self, img_features, txt_features):
        matched = self.pairwise_sampling(img_features, txt_features)

        distances = torch.cdist(img_features, txt_features, p=2)

        intra_loss = torch.sum(distances * matched)
        not_matched = 1 - matched
        inter_loss = torch.sum((1 / (distances + 1e-6)) * not_matched)

        total_loss = intra_loss + self.lambda_ * inter_loss
        if self.reduction == 'mean':
            total_loss /= matched.numel()
        return total_loss
