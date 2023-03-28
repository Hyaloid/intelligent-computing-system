# coding=utf-8
import numpy as np


class FullyConnectedLayer:
    def __init__(self, num_input, num_output):
        self.d_bias = None
        self.d_weight = None
        self.output = None
        self._input = None
        self.bias = None
        self.weight = None
        self.num_input = num_input
        self.num_output = num_output

    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])

    def forward(self, _input):
        self._input = _input
        self.output = self._input.dot(self.weight) + self.bias
        return self.output

    def backward(self, top_diff):
        # 全连接层的反向传播，计算参数梯度和本层损失
        self.d_weight = np.matmul(self._input.T, top_diff)
        self.d_bias = np.matmul(np.ones([1, top_diff.shape[0]]), top_diff)
        bottom_diff = np.matmul(top_diff, self.weight.T)
        return bottom_diff

    def update_param(self, lr):
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias

    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

    def save_param(self):
        return self.weight, self.bias


class ReLULayer:
    def __init__(self):
        self._input = None

    def forward(self, _input):
        self._input = _input
        output = np.maximum(0, self._input)
        return output

    def backward(self, top_diff):
        bottom_diff = top_diff * (self._input >= 0.)
        return bottom_diff


class SoftmaxLossLayer:
    def __init__(self):
        self.prob = None
        self.label_onehot = None
        self.batch_size = None

    def forward(self, _input):
        input_max = np.max(_input, axis=1, keepdims=True)
        input_exp = np.exp(_input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    def get_loss(self, label):
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss

    def backward(self):
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff

