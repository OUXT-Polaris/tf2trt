#!/bin/bash
set -Ceu

cmake.
make

./infer ../resnet_v1_50_finetuned_4class_altered_model.plan /home/ubuntu/tensorrt-ros-node/catkin_ws/src/hoge/data/o0.png map

./infer ../resnet_v1_50_finetuned_4class_altered_model.plan /home/ubuntu/tensorrt-ros-node/catkin_ws/src/hoge/data/o0.png separate

