#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import
import numpy as np
import keras
import tensorflow as tf
import resnet_v1
import cv2
import matplotlib.pyplot as plt
slim = tf.contrib.slim

# resnetの前処理、後処理のコード
# https://github.com/NVIDIA-Jetson/tf_to_trt_image_classification/blob/master/scripts/model_meta.py を参考にした
def preprocess_vgg(image):
    return np.array(image, dtype=np.float32) - np.array([123.68, 116.78, 103.94])

# グラフの初期化Variable v1/weights already exists,reuse=True or reuse=tf.AUTO_REUS 対策
tf.reset_default_graph()

# モデル定義
# https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c/
images = tf.placeholder(tf.float32, (None, 224, 224, 3), name='images')
labels = tf.placeholder(tf.int32, (None, 4), name='labels')
is_training = tf.placeholder(tf.bool)
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, num_classes=4)

# 全体をrestoreするときに使うやつ
restorer = tf.train.Saver()

with tf.Session() as sess:
    # 初期化
    # restorer.restore(sess, "../weights/resnet_v1_50_finetuned_4class_altered_model.ckpt") # 全体を復元する場合
    restorer.restore(sess, "../weights/resnet_v1_50_ft_double_longer_1022.ckpt") # 全体を復元する場合

    # imageの準備
    import glob
    for f in glob.glob('../../OUXT_imageData/datasets/dataset/other/*.png'):
        image = cv2.imread(f)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = preprocess_vgg(image)
        # plt.imshow(image), plt.show()

        # is_training=Falseを入れないとdropoutやらが効いて結果がおかしくなる。推論時はFalseをfeedする
        out = sess.run([logits], feed_dict={images: image[None, ...], is_training: False})[0]
        print(f, out[0], out[0].argmax())

