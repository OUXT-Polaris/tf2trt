#!/bin/bash
set -Ceu

# 一度にrunするスクリプト
# jetson tx2上で動作確認
#
# bash convert.sh weights/resnet_v1_50.ckpt
# @param ckpt: tensorflowで学習済みのResNet重みデータ

if [ $# -ne 2 ]; then
  echo "hoge"
fi

PROJECT_NAME=$(basename $1 .ckpt)
DIR_NAME=$(dirname $1)

CKPT_FILE=$1
PB_FILE=${DIR_NAME}/${PROJECT_NAME}.pb
UFF_FILE=${DIR_NAME}/${PROJECT_NAME}.uff
PLAN_FILE=${DIR_NAME}/${PROJECT_NAME}.plan

IN_NAME='images'
OUT_NAME='resnet_v1_50/SpatialSqueeze'

echo "(input) $1"
echo "(output) ${PLAN_FILE}"

cd train_convert/
python tf_2_uff.py ../$CKPT_FILE ../$PB_FILE ../$UFF_FILE

cd ..
cd plan/
cmake .
make
./plan ../${UFF_FILE} ../${PLAN_FILE} $IN_NAME $OUT_NAME

echo "(exit)"

# cd ..
# cd infer/
# cmake .
# make
# ./infer ../${PLAN_FILE} ${TEST_IMAGE} map_false

