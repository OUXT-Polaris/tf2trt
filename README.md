# Jetson TX2, TensorRTで動かす

## caveat
- ResNetのpretrainedな重みをダウンロードしておく。
    weights/に http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz をダウンロードして解凍
    https://github.com/NVIDIA-Jetson/tf_to_trt_image_classification/blob/master/scripts/download_models.sh 参照
- 各ソース内の`project_name`に従ってweights/以下の重みを読み込み・保存するように動く。動かしたい全部で一致させる。

## 1. TensorflowでResNetを転移学習 fine tuningする(python)
ckptファイルを出力する。 datasetは適宜指定する。kerasで流す。
ジョブバッチシステムを持ってるサーバーで学習させるときはrun.shが使える。
```
cd train_convert
python finetune_tf.py
or
qsub run.sh
```

学習状況・モデルの詳細は以下のコマンドでtensorboard上で見れます。 http://localhost:6006
```
tensorboard --logdir=weights/"$project_name"_summary/
```

## 2. TensorRTで実行可能なUFFに変換する(python)
tensorflowで作ったグラフ、学習したckptファイルから、frozen_graph(.pbファイル)とUFFファイルを作る
TensorRTのインストールされた環境下で、下のリンクに従ってtensorrtのuff python bindingをインストールする
https://github.com/NVIDIA-Jetson/tf_to_trt_image_classification#setup
jetson tx2か、tensorrtの入ってるnvidia gpuの環境でしか動かないと思う。
```
cd train_convert
python tf_2_uff.py
```

## 3. UFFがパースできるか？、そしてPLAN形式にして保存(c++)
uff -> planの変換。
実際に推論させる環境(jetson)上でやらないとだめ。.planのバイナリが出てくる。
```
cd plan
cmake .
make
```

## 4. 画像をロードしてGPUで推論するサンプル
infer/infer.cu
10回推論して、それぞれ掛かった時間を出力する。テストイメージは固定で、preprocessingの時間は含まない。
```
cd infer
cmake .
make
```
