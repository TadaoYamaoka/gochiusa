#!/usr/bin/python
import numpy as np
import chainer
from chainer import cuda, Function, Variable
from chainer import optimizers, serializers
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L

from xml.etree.ElementTree import *
import os
from PIL import Image
from PIL import ImageOps
import math
import sys
import argparse
from datetime import datetime

import random
import cv2
from matplotlib import pylab as plt

from facial_landmark import *

parser = argparse.ArgumentParser(description='Train facial landmark')
parser.add_argument('xmlfile', type=str, help='xmlfile created by imglab tool')
parser.add_argument('testfile', type=str, help='xmlfile created by imglab tool')
parser.add_argument('--batchsize', '-b', type=int, default=16, help='Number of images in each mini-batch')
parser.add_argument('--iteration', '-i', type=int, default=100, help='Number of iteration times')
parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epoch times')
parser.add_argument('--initmodel', '-m', default='', help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='', help='Resume the optimization from snapshot')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
args = parser.parse_args()

model = MyChain()

model.to_gpu()

#optimizer = optimizers.SGD(lr=args.lr)
optimizer = optimizers.RMSprop(lr=args.lr)
#optimizer = optimizers.Adam()
optimizer.setup(model)
#optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

# 訓練データ読み込み
train_data = []
read_image_data(args.xmlfile, train_data)

# テストデータ読み込み
test_data = []
read_image_data(args.testfile, test_data)

#import matplotlib.pyplot as plt
#data = train_data[0]
#img = data[0]
#parts = data[1]
#print(parts)
#plt.imshow(img, 'gray')
#plt.show()

#for i in range(len(train_data)):
#    data = data_augmentation(train_data[i])
#
#    print(i)
#    print(data['parts'])

def mini_batch_data(train_data):
    img_data = []
    t_data = []
    for j in range(args.batchsize):
        data = data_augmentation(train_data[random.randrange(len(train_data))])
        img_data.append(data['img'])
        t_data.append(data['parts'])
        # for debug
        #show_img_and_landmark(data['img'], data['parts'])

    x = Variable(cuda.to_gpu(np.array(img_data, dtype=np.float32)))
    x = F.reshape(x, (args.batchsize, 1, imgsize, imgsize))

    t = Variable(cuda.to_gpu(np.array(t_data, dtype=np.float32)))
    t = F.reshape(t, (args.batchsize, landmark*2))

    return x, t

def save_model(sufix = ""):
    # Save the model and the optimizer
    print('save the model')
    serializers.save_npz('model' + sufix, model)
    print('save the optimizer')
    serializers.save_npz('state' + sufix, optimizer)

print("{}, start training".format(datetime.now()))

for epo in range(args.epoch):
    sum_loss = 0

    for i in range(args.iteration):
        # ミニバッチ入力データ
        x, t = mini_batch_data(train_data)

        # 順伝播
        y = model(x)

        # 誤差逆伝播
        model.cleargrads()
        loss = F.mean_squared_error(y, t)
        loss.backward()
        optimizer.update()

        # loss
        sum_loss += float(loss.data)

        # for debug
        #print(loss.data)
        #print(y.data)
        #print(t.data)


    # 検証
    # ミニバッチ入力データ
    x, t = mini_batch_data(test_data)
    # 順伝播
    y = model(x)
    loss = F.mean_squared_error(y, t)

    print("{}, epoch = {}, train loss = {}, test loss = {}".format(datetime.now(), epo+1, sum_loss / args.iteration, loss.data), flush=True)

    # Save the model and the optimizer
    if (epo+1) % 10000 == 0 or epo == args.epoch - 1:
        save_model(str(epo+1))

    # for debug
    #show_img_and_landmark(cuda.to_cpu(x.data)[0][0], cuda.to_cpu(y.data)[0].reshape((landmark,2)))
