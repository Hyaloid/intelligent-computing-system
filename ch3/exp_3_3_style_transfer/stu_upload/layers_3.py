# coding:utf-8
import numpy as np
import struct
import os
import scipy.io
import time


class ContentLossLayer:
    def __init__(self):
        print('\tContent loss layer.')

    def forward(self, input_layer, content_layer):
        # TODO： 计算风格迁移图像和目标内容图像的内容损失
        n, c, h, w = input_layer.shape[0], input_layer.shape[1], input_layer.shape[2], input_layer.shape[3]
        loss = np.sum((input_layer - content_layer) ** 2) / (2 * n * c * h * w)
        return loss

    def backward(self, input_layer, content_layer):
        # TODO： 计算内容损失的反向传播
        n, c, h, w = input_layer.shape[0], input_layer.shape[1], input_layer.shape[2], input_layer.shape[3]
        bottom_diff = (input_layer - content_layer) / (n * c * h * w)
        return bottom_diff


class StyleLossLayer:
    def __init__(self):
        self.input_layer_reshape = None
        self.gram_style = None
        self.gram_input = None
        self.div = None
        print('\tStyle loss layer.')

    def forward(self, input_layer, style_layer):
        # TODO： 计算风格迁移图像和目标风格图像的 Gram 矩阵
        # style_layer_reshape = np.reshape(style_layer, [style_layer.shape[0], style_layer.shape[1], -1])
        self.gram_style = np.zeros([style_layer.shape[0], style_layer.shape[1], style_layer.shape[1]])
        self.input_layer_reshape = np.reshape(input_layer, [input_layer.shape[0], input_layer.shape[1], -1])
        self.gram_input = np.zeros([input_layer.shape[0], input_layer.shape[1], input_layer.shape[1]])
        for idxn in range(input_layer.shape[0]):
            self.gram_input[idxn, :, :] = np.matmul(self.input_layer_reshape[idxn, :, :], self.input_layer_reshape[idxn, :, :].T)

        M = input_layer.shape[2] * input_layer.shape[3]
        N = input_layer.shape[1]
        self.div = M * M * N * N
        # TODO： 计算风格迁移图像和目标风格图像的风格损失
        style_diff = np.sum((self.gram_style - self.gram_input) ** 2)
        loss = style_diff / (4 * input_layer.shape[0] * self.div)
        return loss

    def backward(self, input_layer, style_layer):
        bottom_diff = np.zeros(
            [input_layer.shape[0], input_layer.shape[1], input_layer.shape[2] * input_layer.shape[3]])
        for idxn in range(input_layer.shape[0]):
            # TODO： 计算风格损失的反向传播
            bottom_diff[idxn, :, :] = np.matmul((self.gram_input - self.gram_style)[idxn, :, :].T, style_layer.reshape(style_layer.shape[0], style_layer.shape[1], -1)[idxn, :, :]) / (style_layer.shape[0] * self.div)
        bottom_diff = np.reshape(bottom_diff, input_layer.shape)
        return bottom_diff
