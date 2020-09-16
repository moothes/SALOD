import torch
from torch import nn
import torch.nn.functional as F

class GradLayer(nn.Module):
    def __init__(self):
        super(GradLayer, self).__init__()
        self.grad_x = nn.Conv2d(2, 2, 3, padding=1, bias=False)
        self.grad_y = nn.Conv2d(2, 2, 3, padding=1, bias=False)
        self.set_weight()

    def set_weight(self):
        x = torch.Tensor([[-1., 0, 1], [-2., 0, 2.], [-1., 0, 1.]]).view(1, 1, 3, 3)
        y = torch.Tensor([[-1., -2., -1.], [0, 0, 0], [1., 2., 1.]]).view(1, 1, 3, 3)
        weight_x = nn.Parameter(x, requires_grad=False)
        weight_y = nn.Parameter(y, requires_grad=False)
        self.grad_x.weight, self.grad_y.weight = weight_x, weight_y

    def forward(self, x):
        x1, x2 = self.grad_x(x), self.grad_y(x)
        # return torch.sqrt(torch.pow(x1, 2) + torch.pow(x2, 2))
        return torch.pow(x1, 2) + torch.pow(x2, 2)


def Loss(X, batchs, config):
    pred = X['final'].sigmoid_()

    loss = F.binary_cross_entropy(pred, batchs)
    area_loss = 1 - 2 * ((pred * batchs).sum() + 1) / (pred.sum() + batchs.sum() + 1)
    loss += area_loss
    
    return loss
    
class Loss_orig(nn.Module):
    def __init__(self, area=True, boundary=False, contour_th=1.5, ratio=1):
        super(Loss_orig, self).__init__()
        self.area, self.boundary, self.cth, self.ratio = area, boundary, contour_th, ratio
        if boundary:
            self.gradlayer = GradLayer()

    def forward(self, x, label):
        loss = F.binary_cross_entropy(x, label)
        if self.area:
            area_loss = 1 - 2 * ((x * label).sum() + 1) / (x.sum() + label.sum() + 1)
            loss += area_loss
        if self.boundary:
            prob_grad = F.tanh(self.gradlayer(x))
            label_grad = torch.gt(self.gradlayer(label), self.cth).float()
            inter = torch.sum(prob_grad * label_grad)
            union = torch.pow(prob_grad, 2).sum() + torch.pow(label_grad, 2).sum()
            boundary_loss = (1 - 2 * (inter + 1) / (union + 1))
            loss = loss + self.ratio * boundary_loss
        return loss