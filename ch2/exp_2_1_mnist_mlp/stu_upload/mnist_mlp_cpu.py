# coding=utf-8
import os
import struct
import numpy as np
from layers_1 import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer


# for local
os.chdir('E:\PyCharmProjects\intelligent-computing-system\ch2')
MNIST_DIR = './data/MNIST/raw'
TRAIN_DATA = 'train/train-images-idx3-ubyte'
TRAIN_LABELS = 'train/train-labels-idx1-ubyte'
TEST_DATA = 'test/t10k-images-idx3-ubyte'
TEST_LABELS = 'test/t10k-labels-idx1-ubyte'

# for upload
# MNIST_DIR = "../mnist_data"
# TRAIN_DATA = "train-images-idx3-ubyte"
# TRAIN_LABELS = "train-labels-idx1-ubyte"
# TEST_DATA = "t10k-images-idx3-ubyte"
# TEST_LABELS = "t10k-labels-idx1-ubyte"


class MNIST_MLP:
    def __init__(
            self,
            batch_size=100,
            input_size=784,
            hidden1=32,
            hidden2=16,
            out_classes=10,
            lr=0.01,
            max_epoch=2,
            print_iter=100
    ):
        """神经网络初始化"""
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter
        self.train_data = None
        self.test_data = None

    @classmethod
    def load_mnist(cls, file_dir, is_images=True):
        """MNIST 数据集文件的读取和预处理"""
        bin_file = open(file_dir, 'rb')
        bin_data = bin_file.read()
        bin_file.close()
        if is_images:
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(
                fmt_header,
                bin_data,
                0
            )
        else:
            fmt_header = '>ii'
            magic, num_images = struct.unpack_from(
                fmt_header,
                bin_data,
                0
            )
            num_rows, num_cols = 1, 1
        data_size = num_images * num_rows * num_cols
        mat_data = struct.unpack_from(
            '>' + str(data_size) + 'B',
            bin_data,
            struct.calcsize(fmt_header)
        )
        mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
        return mat_data

    def load_data(self):
        train_images = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
        train_labels = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_LABELS), False)
        test_images = self.load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
        test_labels = self.load_mnist(os.path.join(MNIST_DIR, TEST_LABELS), False)

        self.train_data = np.append(train_images, train_labels, axis=1)
        self.test_data = np.append(test_images, test_labels, axis=1)
        print(self.train_data, '\n', self.test_data)

    def build_model(self):
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(self.hidden1, self.hidden2)
        self.relu2 = ReLULayer()
        self.fc3 = FullyConnectedLayer(self.hidden2, self.out_classes)
        self.softmax = SoftmaxLossLayer()
        self.update_layer_list = [self.fc1, self.fc2, self.fc3]

    def init_model(self):
        for layer in self.update_layer_list:
            layer.init_param()

    def forward(self, _input):
        h1 = self.fc1.forward(_input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        h2 = self.relu2.forward(h2)
        h3 = self.fc3.forward(h2)
        prob = self.softmax.forward(h3)
        return prob

    def backward(self):
        dloss = self.softmax.backward()
        dh3 = self.fc3.backward(dloss)
        dh2 = self.relu2.backward(dh3)
        dh2 = self.fc2.backward(dh2)
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def update(self, lr):
        for layer in self.update_layer_list:
            layer.update_param(lr)

    def save_model(self, param_dir):
        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        params['w3'], params['b3'] = self.fc3.save_param()
        np.save(param_dir, params)

    def load_model(self, param_dir):
        params = np.load(param_dir, allow_pickle=True).item()
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])
        self.fc3.load_param(params['w3'], params['b3'])

    def shuffle_data(self):
        print('shuffle data...')
        np.random.shuffle(self.train_data)

    def train(self):
        max_batch = self.train_data.shape[0] / self.batch_size
        for idx_epoch in range(self.max_epoch):
            self.shuffle_data()
            for idx_batch in range(int(max_batch)):
                batch_images = self.train_data[idx_batch * self.batch_size: (idx_batch + 1) * self.batch_size, :-1]
                batch_labels = self.train_data[idx_batch * self.batch_size: (idx_batch + 1) * self.batch_size, -1]

                prob = self.forward(batch_images)
                loss = self.softmax.get_loss(batch_labels)
                self.backward()
                self.update(self.lr)
                if idx_batch % self.print_iter == 0:
                    print('Epoch %d, iter %d, loss: %.6f' % (idx_epoch, idx_batch, loss))

    def evaluate(self):
        pred_results = np.zeros([self.test_data.shape[0]])
        for idx in range(int(self.test_data.shape[0] / self.batch_size)):
            batch_images = self.test_data[idx * self.batch_size: (idx + 1) * self.batch_size, :-1]
            prob = self.forward(batch_images)
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx * self.batch_size: (idx + 1) * self.batch_size] = pred_labels
        accuracy = np.mean(pred_results == self.test_data[:, -1])
        print('Accuracy in test set: %f' % accuracy)


def build_mnist_mlp(param_dir='weight.npy'):
    h1, h2, e = 1024, 256, 10
    mlp = MNIST_MLP(hidden1=h1, hidden2=h2, max_epoch=e)
    mlp.load_data()
    mlp.build_model()
    mlp.init_model()
    mlp.train()
    mlp.save_model('mlp-%d-%d-%depoch.npy' % (h1, h2, e))
    return mlp


if __name__ == '__main__':
    mlp = build_mnist_mlp()
    mlp.evaluate()

