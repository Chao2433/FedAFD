from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        positive_dist = torch.pow(anchor - positive, 2).sum(1)  # [N, 1]
        negative_dist = torch.pow(anchor - negative, 2).sum(1)  # [N, 1]
        y = torch.ones_like(positive_dist)  # y == 1
        # loss = self.ranking_loss(positive_dist, negative_dist, y)
        loss = self.ranking_loss(negative_dist, positive_dist, y)
        return loss


class TripletLoss_uni(nn.Module):
    def __init__(self, margin=0.2, hard_negative=False):
        super(TripletLoss_uni, self).__init__()
        self.margin = margin
        self.hard_negative = hard_negative

    def forward(self, ie, te):
        scores = ie.mm(te.t())
        diagonal = scores.diag().view(ie.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_i = (self.margin + scores - d1).clamp(min=0)

        cost_t = (self.margin + scores - d2).clamp(min=0)

        mask = torch.eye(scores.size(0), dtype=torch.bool)
        I = torch.autograd.Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()

        cost_i = cost_i.masked_fill_(I, 0)
        cost_t = cost_t.masked_fill_(I, 0)

        if self.hard_negative:
            cost_i = cost_i.max(1)[0]
            cost_t = cost_t.max(0)[0]

        return cost_t.sum() + cost_i.sum()





