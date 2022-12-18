import torch
from torch.nn import functional as F
from torch.nn.modules import Module

from utils import pairwise_distances


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''

    def __init__(self):
        super(PrototypicalLoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, input, target, n_support):
        return prototypical_loss(input, target, n_support, self.device)


def prototypical_loss(input, target, n_support, device):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''

    classes = torch.unique(target)
    n_classes = len(classes)

    # Make prototypes
    support_idxs = torch.stack(list(map(lambda c: target.eq(c).nonzero()[:n_support].squeeze(1), classes)))
    prototypes = torch.stack([input[idx_list].mean(0) for idx_list in support_idxs])

    # Make query samples
    n_query = target.eq(classes[0].item()).sum().item() - n_support
    query_idxs = torch.stack(list(map(lambda c: target.eq(c).nonzero()[n_support:], classes))).view(-1)
    query_samples = input[query_idxs]

    dists = pairwise_distances(query_samples, prototypes, 'l2')

    log_p_y = F.log_softmax(-dists, dim=1)
    y_hat = log_p_y.argmax(1)
    target_label = torch.arange(0, n_classes, 1 / n_query).long().to(device)

    loss_val = torch.nn.NLLLoss()(log_p_y, target_label)
    acc_val = y_hat.eq(target_label.squeeze()).float().mean()

    return loss_val, acc_val