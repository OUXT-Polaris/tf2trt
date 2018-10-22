# -*- coding: utf-8 -*-
"""
tensorflowで作ったグラフ、学習したckptファイルから、frozen_graph(.pbファイル)とUFFファイルを作る
TensorRTのインストールされた環境下で、下のリンクに従ってtensorrtのuff python bindingをインストールする
https://github.com/NVIDIA-Jetson/tf_to_trt_image_classification#setup
"""
import tensorflow as tf
import tensorflow.contrib.slim as tf_slim
import uff
import resnet_v1
slim = tf.contrib.slim

import sys
if len(sys.argv) == 4:
    print(sys.argv)
    IN = sys.argv[1]
    PB = sys.argv[2]
    OUT = sys.argv[3]
    print(IN, PB, OUT)
    exit(0)
else:
    exit(0)

# 初期設定
tf.reset_default_graph()
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config=tf_config)

# モデルの構築
images = tf.placeholder(tf.float32, (None, 224, 224, 3), name='images')
labels = tf.placeholder(tf.int32, (None, 1, 1, 4), name='labels')
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    logits, end_points = resnet_v1.resnet_v1_50(images, is_training=False, num_classes=4)

# チェックポイントの読み込み
saver = tf.train.Saver()
saver.restore(save_path=IN, sess=sess)

# freeze graph
output_nodes=['resnet_v1_50/SpatialSqueeze']
frozen_graph = tf.graph_util.convert_variables_to_constants(
        sess, 
        sess.graph.as_graph_def(), 
        output_node_names=output_nodes)
from convert_relu6 import convertRelu6
frozen_graph = convertRelu6(frozen_graph)

# pbとして保存
with open(PB, 'wb') as f:
    f.write(frozen_graph.SerializeToString())

# uffに変換
uff_model = uff.from_tensorflow_frozen_model(
        frozen_file=PB,
        output_nodes=output_nodes,
        output_filename=OUT,
        text=False)
print('finished', PB)
