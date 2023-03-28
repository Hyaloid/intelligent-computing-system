# coding:utf-8
from stu_upload.vgg19_demo import VGG19
import time
import numpy as np
import os
import scipy.io

<<<<<<< HEAD
=======

>>>>>>> e4a09f764d67be67e0fb1f85c007dc91a2ad3b7e
def evaluate(vgg):
    start = time.time()
    vgg.forward()
    end = time.time()
<<<<<<< HEAD
    print('inference time: %f'%(end - start))
    result = vgg.net.getOutputData()
    prob = max(result)
    top1 = result.index(prob)
    print('Classification result: id = %d, prob = %f'%(top1, prob))
=======
    print('inference time: %f' % (end - start))
    result = vgg.net.getOutputData()
    prob = max(result)
    top1 = result.index(prob)
    print('Classification result: id = %d, prob = %f' % (top1, prob))
>>>>>>> e4a09f764d67be67e0fb1f85c007dc91a2ad3b7e


if __name__ == '__main__':
    vgg = VGG19()
<<<<<<< HEAD
    vgg.build_model(param_path='../imagenet-vgg-verydeep-19.mat', 
=======
    vgg.build_model(param_path='../imagenet-vgg-verydeep-19.mat',
>>>>>>> e4a09f764d67be67e0fb1f85c007dc91a2ad3b7e
                    quant_param_path='../vgg19_quant_param_new.npz')
    vgg.load_model()
    vgg.load_image('../cat1.jpg')
    for i in range(10):
<<<<<<< HEAD
        evaluate(vgg)
=======
        evaluate(vgg)
>>>>>>> e4a09f764d67be67e0fb1f85c007dc91a2ad3b7e
