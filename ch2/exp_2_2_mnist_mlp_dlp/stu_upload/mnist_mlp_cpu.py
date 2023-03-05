# coding=utf-8
from ch2.exp_2_1_mnist_mlp.stu_upload.mnist_mlp_cpu import MNIST_MLP

# for local
# MNIST_DIR = '../data/MNIST/raw'
# TRAIN_DATA = 'train/train-images-idx3-ubyte'
# TRAIN_LABELS = 'train/train-labels-idx1-ubyte'
# TEST_DATA = 'test/t10k-images-idx3-ubyte'
# TEST_LABELS = 'test/t10k-labels-idx1-ubyte'

# for upload
MNIST_DIR = "../mnist_data"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABELS = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABELS = "t10k-labels-idx1-ubyte"


def build_mnist_mlp(param_dir='weight.npy'):
    h1, h2, e = 1024, 256, 10
    mlp = MNIST_MLP(hidden1=h1, hidden2=h2, max_epoch=e)
    mlp.load_data()
    mlp.build_model()
    mlp.init_model()
    mlp.train()
    mlp.save_model('./stu_upload/weight.npy')
    return mlp


if __name__ == '__main__':
    mlp = build_mnist_mlp()
    mlp.evaluate()

