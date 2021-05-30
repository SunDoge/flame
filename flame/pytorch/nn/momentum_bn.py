"""
https://github.com/zengarden/momentum2-teacher
"""

from torch import nn, Tensor
import torch
import math


class MomentumBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=1.0, affine=True, track_running_stats=True, total_iters=100):
        super(MomentumBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.total_iters = total_iters
        self.cur_iter = 0
        self.mean_last_batch = None
        self.var_last_batch = None

    def momentum_cosine_decay(self):
        self.cur_iter += 1
        self.momentum = (
            math.cos(math.pi * (self.cur_iter / self.total_iters)) + 1) * 0.5

    def forward(self, x):
        # if not self.training:
        #     return super().forward(x)

        mean = torch.mean(x, dim=[0, 2, 3])
        var = torch.var(x, dim=[0, 2, 3])
        n = x.numel() / x.size(1)

        with torch.no_grad():
            tmp_running_mean = self.momentum * mean + \
                (1 - self.momentum) * self.running_mean
            # update running_var with unbiased var
            tmp_running_var = self.momentum * var * n / \
                (n - 1) + (1 - self.momentum) * self.running_var

        x = (x - tmp_running_mean[None, :, None, None].detach()) / (
            torch.sqrt(tmp_running_var[None, :,
                       None, None].detach() + self.eps)
        )
        if self.affine:
            x = x * self.weight[None, :, None, None] + \
                self.bias[None, :, None, None]

        # update the parameters
        if self.mean_last_batch is None and self.var_last_batch is None:
            self.mean_last_batch = mean
            self.var_last_batch = var
        else:
            self.running_mean = (
                self.momentum * ((mean + self.mean_last_batch) * 0.5) +
                (1 - self.momentum) * self.running_mean
            )
            self.running_var = (
                self.momentum * ((var + self.var_last_batch)
                                 * 0.5) * n / (n - 1)
                + (1 - self.momentum) * self.running_var
            )
            self.mean_last_batch = None
            self.var_last_batch = None
            self.momentum_cosine_decay()

        return x


class MomentumBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=1.0, affine=True, track_running_stats=True, total_iters=100):
        super(MomentumBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.total_iters = total_iters
        self.cur_iter = 0
        self.mean_last_batch = None
        self.var_last_batch = None

    def momentum_cosine_decay(self):
        self.cur_iter += 1
        self.momentum = (
            math.cos(math.pi * (self.cur_iter / self.total_iters)) + 1) * 0.5

    def forward(self, x):
        # if not self.training:
        #     return super().forward(x)

        mean = torch.mean(x, dim=[0])
        var = torch.var(x, dim=[0])
        n = x.numel() / x.size(1)

        with torch.no_grad():
            tmp_running_mean = self.momentum * mean + \
                (1 - self.momentum) * self.running_mean
            # update running_var with unbiased var
            tmp_running_var = self.momentum * var * n / \
                (n - 1) + (1 - self.momentum) * self.running_var

        x = (x - tmp_running_mean[None, :].detach()) / \
            (torch.sqrt(tmp_running_var[None, :].detach() + self.eps))
        if self.affine:
            x = x * self.weight[None, :] + self.bias[None, :]

        # update the parameters
        if self.mean_last_batch is None and self.var_last_batch is None:
            self.mean_last_batch = mean
            self.var_last_batch = var
        else:
            self.running_mean = (
                self.momentum * ((mean + self.mean_last_batch) * 0.5) +
                (1 - self.momentum) * self.running_mean
            )
            self.running_var = (
                self.momentum * ((var + self.var_last_batch)
                                 * 0.5) * n / (n - 1)
                + (1 - self.momentum) * self.running_var
            )
            self.mean_last_batch = None
            self.var_last_batch = None
            self.momentum_cosine_decay()

        return x
