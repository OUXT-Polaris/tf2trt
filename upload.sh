#!/bin/bash
set -Ceu

# s3にplanファイルをアップロードする
aws s3 cp weights/resnet_v1_50_ft_double_longer_1022.plan s3://cnn-prediction-weights/ --acl public-read
aws s3 cp weights/resnet_v1_50_ft_double_longer_1022.uff  s3://cnn-prediction-weights/ --acl public-read

