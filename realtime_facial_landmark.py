#!/usr/bin/python
import numpy as np
import chainer
from chainer import Function, Variable
from chainer import serializers
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L

import os
import math
import sys
import argparse

import random
import cv2
import dlib

from facial_landmark import *

parser = argparse.ArgumentParser(description='Realtime predict facial landmark')
parser.add_argument('model', type=str, help='model file')
args = parser.parse_args()

model = MyChain()
serializers.load_npz(args.model, model)

detector = dlib.simple_object_detector("detector.svm")

video_capture = cv2.VideoCapture(1)

while True:
    ret, frame = video_capture.read()

    # 顔検出
    dets = detector(frame)

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔領域を切り出してミニバッチに追加
    img_data = []
    matrix = []
    for d in dets:
        w = d.right() - d.left()
        h = d.bottom() - d.top();
        if h > w:
            w = h
        scale = 100.0 / w
        M = np.array([[scale, 0, -d.left() * scale], [0, scale, -d.top() * scale]])
        dst = cv2.warpAffine(gray, M, (100, 100))
        cropped = 1.0 - np.array(dst, dtype=np.float32) / 256.0
        cv2.imshow('cropped', dst)
        img_data.append(cropped)
        matrix.append(M)

    # 顔器官検出
    if len(img_data) > 0:
        x = Variable(np.array(img_data, dtype=np.float32))
        x = F.reshape(x, (len(img_data), 1, imgsize, imgsize))
        y = model(x)

        for i, d in enumerate(dets):
            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))
            q = y.data[i].reshape(landmark, 2) * 100
            # 元の座標に変換
            M = matrix[i]
            invMT = cv2.invertAffineTransform(M).T
            for k in range(landmark):
                p = np.append(q[k], 1).dot(invMT)
                cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 0, 255), 1)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()