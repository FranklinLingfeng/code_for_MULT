import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd

class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, momentum_fix, features, momentum, threshold=0.5):
        
        ctx.threshold = threshold
        
        ctx.features = features
        ctx.momentum = momentum
        ctx.momentum_fix = momentum_fix
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        if targets is not None:
            if ctx.momentum_fix == True:
                for x, y in zip(inputs, targets):
                    ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
                    ctx.features[y] /= ctx.features[y].norm()
            # else:
            #     for x, y in zip(inputs, targets):
            #         idx = y.argmax()
            #         ctx.features[idx] = (1. - (1. - ctx.momentum) * y.max()) * ctx.features[idx] + (1. - ctx.momentum) * y.max() * x
            #         ctx.features[idx] /= ctx.features[idx].norm()

        return grad_inputs, None, None, None, None, None, None


def cm(inputs, indexes, momentum_fix, features, momentum=0.5, threshold=0.5):
    return CM.apply(inputs, indexes, momentum_fix, features, torch.Tensor([momentum]).to(inputs.device), threshold)


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.threshold = 0.5

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets=None, momentum_fix=True):

        inputs = F.normalize(inputs, dim=1)
        
        outputs = cm(inputs, targets, momentum_fix, self.features, self.momentum, threshold=self.threshold)
        outputs /= self.temp

        return outputs
    
