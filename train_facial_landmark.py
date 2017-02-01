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

import random
import cv2
from matplotlib import pylab as plt

parser = argparse.ArgumentParser(description='Train facial landmark')
parser.add_argument('xmlfile', type=str, help='xmlfile created by imglab tool')
parser.add_argument('--imgsize', type=int, default=100, help='Image width and height')
parser.add_argument('--landmark', type=int, default=11, help='Number of landmarks')
parser.add_argument('--batchsize', '-b', type=int, default=16, help='Number of images in each mini-batch')
parser.add_argument('--iteration', '-i', type=int, default=100, help='Number of iteration times')
args = parser.parse_args()


class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Convolution2D(in_channels = 1, out_channels = 16, ksize = 4),
            l2=L.Convolution2D(in_channels = 16, out_channels = 32, ksize = 5),
            l3=L.Convolution2D(in_channels = 32, out_channels = 64, ksize = 5),
            l4=L.Linear(int(((args.imgsize-2)/2-3)/2-4)**2*64, 400),
            l5=L.Linear(400, args.landmark*2)
        )
        
    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.l1(x)), 2)
        h2 = F.max_pooling_2d(F.relu(self.l2(h1)), 2)
        h3 = F.relu(self.l3(h2))
        h3_reshape = F.reshape(h3, (len(h3.data), int(h3.data.size / len(h3.data))))
        h4 = F.relu(self.l4(h3_reshape))
        return self.l5(h4)

model = MyChain()
optimizer = optimizers.SGD()
optimizer.use_cleargrads()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

# 読み込み
tree = parse(args.xmlfile)
root = tree.getroot()
images = root.find("images")

def get_distance(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.sqrt(dx * dx + dy * dy)

def get_center(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return [a[0] + dx / 2, a[1] + dy / 2]

def get_box_from_center_and_length(c, l):
    return (c[0] - l, c[1] - l, c[0] + l, c[1] + l)

train_data = []

for image in list(images):
    img = ImageOps.invert(Image.open(image.get("file")).convert('L'))
    for box in list(image):
        parts = {}
        for part in list(box):
            parts[part.get("name")] = [int(part.get("x")), int(part.get("y")), 1]

        if "L01" not in parts or "R02" not in parts:
            print("parts not found. " + image.get("file"))
            continue

        c = get_center(parts["L01"], parts["R02"])
        l = get_distance(c, parts["C02"]) * 1.7
        
        # 画像切り取りグレイスケールの値の範囲を0から1にする
        cropped = np.array(img.crop(get_box_from_center_and_length(c, l)), dtype=np.float32) / 256.0

        # ランドマークの座標変換
        for part in parts.values():
            # 左上を0,0にする
            part[0] -= c[0] - l # x
            part[1] -= c[1] - l # y

        train_data.append({'img': cropped, 'parts' : parts})


#import matplotlib.pyplot as plt
#data = train_data[0]
#img = data[0]
#parts = data[1]
#print(parts)
#plt.imshow(img, 'gray')
#plt.show()

def data_augmentation(data):
    img = data['img']
    parts = data['parts']

    width = img.shape[1]
    height = img.shape[0]

    center = (width / 2, height / 2)

    dx = parts["R02"][0] - parts["L01"][0]
    dy = - parts["R02"][1] + parts["L01"][1]
    angle0 = math.degrees(math.atan(dy / dx))

    # 2/3の範囲を100*100にする
    scale0 = 100.0 / (width * 2.0 / 3.0)

    # ランダムに変形を加える
    angle = random.uniform(-45, 45) - angle0
    scale = scale0 * random.uniform(0.9, 1.1)

    # 変形後の原点
    #x0 = width / 3.0 * scale

    # アフィン変換
    #  回転、拡大の後に、回転の中心が(50, 50)になるように平行移動
    matrix = cv2.getRotationMatrix2D(center, angle, scale) + np.array([[0, 0, -center[0] + 50], [0, 0, -center[1] + 50]])
    dst = cv2.warpAffine(img, matrix, (100, 100))

    # ランドマーク座標をnumpyの配列に変換
    parts_np = np.array([
        parts["C01"], parts["C02"], parts["C03"], parts["L01"], parts["L02"], parts["L03"], parts["M01"], parts["M02"], parts["R01"], parts["R02"], parts["R03"]], dtype=np.float32)

    # ランドマークの座標変換
    parts_converted = parts_np.dot(matrix.T) / 100.0

    # 変換されたデータを返す
    return {'img' : dst, 'parts' : parts_converted}


#for i in range(len(train_data)):
#    data = data_augmentation(train_data[i])
#
#    print(i)
#    print(data['parts'])

def show_img_and_landmark(img, parts):
    plt.imshow(1.0 - img, cmap='gray')
    for t in parts:
        plt.plot(t[0]*100, t[1]*100, '.r')
    plt.axis([0, 100, 100, 0])
    plt.show()

def mini_batch_data(train_data):
    img_data = []
    t_data = []
    for j in range(args.batchsize):
        data = data_augmentation(train_data[random.randrange(len(train_data))])
        img_data.append(data['img'])
        t_data.append(data['parts'])
        # for debug
        show_img_and_landmark(data['img'], data['parts'])

    x = Variable(np.array(img_data, dtype=np.float32))
    x = F.reshape(x, (args.batchsize, 1, args.imgsize, args.imgsize))

    t = Variable(np.array(t_data, dtype=np.float32))
    t = F.reshape(t, (args.batchsize, args.landmark*2))

    return x, t

for i in range(args.iteration):
    # ミニバッチ入力データ
    x, t = mini_batch_data(train_data)

    # 順伝播
    y = model(x)
    optimizer.update(F.mean_squared_error, y, t)
    # for debug
    print(y.data)
    print(t.data)

# Save the model and the optimizer
print('save the model')
serializers.save_npz('model', model)
print('save the optimizer')
serializers.save_npz('state', optimizer)
