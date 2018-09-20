#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
# https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c/
# https://github.com/NVIDIA-Jetson/tf_to_trt_image_classification/tree/master/scripts
# https://github.com/tensorflow/models/blob/a41f00ac171cf53539b4e2de47f2e15ccb848c90/research/slim/nets/mobilenet_v1_train.py
import numpy as np
import keras
import tensorflow as tf
import resnet_v1
slim = tf.contrib.slim

# 出力される重みデータのファイル名
project_name = 'resnet_v1_50_finetuned_4class_altered_model'

# resnetの前処理、後処理のコード
# https://github.com/NVIDIA-Jetson/tf_to_trt_image_classification/blob/master/scripts/model_meta.py を参考にした
output_names = ['resnet_v1_50/SpatialSqueeze']
def create_label_map(label_file='imagenet_labels_1001.txt'):
    label_map = {}
    with open(label_file, 'r') as f:
        labels = f.readlines()
        for i, label in enumerate(labels):
            label_map[i] = label
    return label_map
IMAGNET2012_LABEL_MAP = create_label_map()
def preprocess_vgg(image):
    return np.array(image, dtype=np.float32) - np.array([123.68, 116.78, 103.94])
def postprocess_vgg(output):
    output = output.flatten()
    predictions_top5 = np.argsort(output)[::-1][0:5]
    labels_top5 = [(IMAGNET2012_LABEL_MAP[p + 1], output[p]) for p in predictions_top5]
    return labels_top5

# dataset flowingはkerasを使う
# batchごとに呼び出す。
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        data_format='channels_last',
        preprocessing_function=preprocess_vgg,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
gen = datagen.flow_from_directory(
        'test_dataset',
        target_size=(224,224),
        color_mode='rgb',
        classes=['green', 'red', 'white', 'other'],
        batch_size=16,
        class_mode='categorical')

# グラフの初期化Variable v1/weights already exists,reuse=True or reuse=tf.AUTO_REUS 対策
tf.reset_default_graph()  

# モデル定義
# https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c/
images = tf.placeholder(tf.float32, (None, 224, 224, 3), name='images')
labels = tf.placeholder(tf.int32, (None, 4), name='labels')
is_training = tf.placeholder(tf.bool)
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, num_classes=4)  
    
# モデルの演算の種類を列挙する
# A = set()
# for item in sess.graph.get_operations():
    # A.add(item.type)


# 全体をrestoreするときに使うやつ
restorer = tf.train.Saver()

# 1. 既存のweightをロードする関数 logitの全結合のW,bは使わない(そもそもサイズが合わない)
# weights/resnet_v1_50.ckptは http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz からDL
# https://github.com/NVIDIA-Jetson/tf_to_trt_image_classification/blob/master/scripts/download_models.sh を参照
variables_to_restore = slim.get_variables_to_restore(exclude=['resnet_v1_50/logits/weights', 'resnet_v1_50/logits/biases'])
init_fn = slim.assign_from_checkpoint_fn('../weights/resnet_v1_50.ckpt', variables_to_restore, ignore_missing_vars=True)
# 2. initialization
logits_variables = slim.get_variables('resnet_v1_50/logits')
logits_init = tf.variables_initializer(logits_variables)


# optimizerの定義
loss = tf.losses.softmax_cross_entropy(labels, logits)
total_loss = tf.losses.get_total_loss(name='total_loss')
opt = tf.train.GradientDescentOptimizer(1e-3).minimize(total_loss, var_list=logits_variables) # 動かせるのはlogitのみ
slim.summaries.add_scalar_summary(total_loss, 'total_loss', 'losses')
tf.summary.scalar('loss', loss)
tf.summary.scalar('total loss', total_loss)

# accuracyの計算
# https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c/
prediction = tf.to_int32(tf.argmax(logits, 1))
answer = tf.to_int32(tf.argmax(labels, 1))
correct_prediction = tf.equal(prediction, answer)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
summary = tf.summary.merge_all()


with tf.Session() as sess:
    # 初期化
    # restorer.restore(sess, "./hoge/resnet_v1_50.ckpt") # 全体を復元する場合
    init_fn(sess)  # 1.
    sess.run(logits_init)  # 2.
    
    # ログを残す
    writer = tf.summary.FileWriter('../weights/' + project_name + '_summary', sess.graph)
    
    # fine tuning
    num_epoch = 20
    num_batches_per_epoch = 100
    for epoch in range(num_epoch):
        i = 0
        for x,y in gen:
            # デバッグ用に画像がきちんと入っているかを確かめる
            debug = False
            if debug:
                for i in range(x.shape[0]):
                    image = x[i]
                    plt.imshow(image)
                    plt.show()
                    out = sess.run([logits], feed_dict={images: image[None, ...], is_training: False})[0] 
                    print(postprocess_vgg(out[0,0,0]), out.shape)
            # 学習
            _, w_summary = sess.run([opt, summary], feed_dict={images: x, labels: y, is_training: True})
            writer.add_summary(w_summary, i + epoch * num_batches_per_epoch)
            i += 1
            # 10 batchに一回、画像のaccuracyを報告
            if i % 10 == 0:
                correct_pred = sess.run(correct_prediction, feed_dict={images: x, labels: y, is_training: False})
                train_acc = float(correct_pred.sum()) / correct_pred.shape[0]
                train_loss = total_loss.eval(feed_dict={images: X, labels: Y, is_training: False})
                print(i, t_acc, train_loss)
            # epochの終了
            if i == num_batches_per_epoch:
                break

    # checkpoint saving
    saver = tf.train.Saver()
    saver.save(sess, '../weights/' + project_name + '.ckpt')

    # 推論してみてチェックするか？
    infer = False
    if infer:
        import cv2
        for epoch in range(1):
            # imageの準備
            image = cv2.imread('00002.png')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image = preprocess_vgg(image)
            # plt.imshow(image), plt.show()

            # is_training=Falseを入れないとdropoutやらが効いて結果がおかしくなる。推論時はFalseをfeedする
            out = sess.run([logits], feed_dict={images: image[None, ...], is_training: False})[0] 
            # print(postprocess_vgg(out[0,0,0]), out.shape)
            print(out)
        
