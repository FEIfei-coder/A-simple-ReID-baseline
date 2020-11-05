from __future__ import absolute_import

def DeepSupervision(criterion, xs, y):
    """
    Args:
        criterion: loss_fuc function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    return loss
