# tf-faster-rcnn学习

源：`git clone https://github.com/endernewton/tf-faster-rcnn.git`
此版本已被放弃，最新的支持多GPU训练的 faster/mask RCNN在：[TensorPack](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN)
此版本基于caffe的[py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)

基于文章：
Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
An Implementation of Faster RCNN with Study for Region Sampling

Note：
1. 已支持 VGG16(`conv5_3`), Resnet V1(`last conv4`)和Mobilenet V1
2. 保留小的proposals(< 16 pixels) ，对小目标有效
3. 以下三条对于Resnets，在fine-tuning时fix了第一个block(一共4个)
4. 只使用`crop_and_resize`来resize RoIs，而不是max-pool
5. 最后的feature maps使用average-pooled来分类和回归
6. 所有的batch normalization参数固定
7. 对于Mobilenets，fine-tuning时固定了前5层，BN参数也固定设置，权重衰减为4e-5

其他特点：
1. 训练时，验证集被用于测试是否过拟合。默认的设置中，测试集被用于验证
2. 由于随机性，不能保证恢复训练时的结果一致。( for resuming training )
3. 此版本会记录训练中的gt boxes，losses，activations和variables，用于tensorboard可视化
## 使用
### 安装
```bash
# 前提
pip install cython opencv-python easydict
cd /home/seucar/Desktop/tf-faster-rcnn
source ~/anaconda3/bin/activate faster-rcnn
cd tf-faster-rcnn/lib
# 修改setpu.py中的sm_75，对应GPU计算能力
vim setup.py
# 编译 Cython 模块
make clean
make
cd ..
# 安装 Python COCO API
cd data
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
cd ../../..
```
### 数据准备--VOC
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
cd /home/seucar/Desktop/tf-faster-rcnn/data
ln -s $VOCdevkit VOCdevkit2007
```
`~/Desktop/tf-faster-rcnn/data$ ln -s /media/seucar/Dataset/VOC/VOCdevkit VOCdevkit2007`

### 训练
预训练模型：[GD网盘](https://drive.google.com/drive/folders/0B1_fAEgxdnvJSmF3YUlZcHFqWTQ)
使用预训练模型的脚本：
将GD网盘下载的压缩包，解压到data目录下
```bash
#!/bin/bash
NET=vgg16
TRAIN_IMDB=voc_2007_trainval
mkdir -p output/${NET}/${TRAIN_IMDB}
cd output/${NET}/${TRAIN_IMDB}
ln -s ../../../data/voc_2007_trainval ./default
cd ../../..
# 在根目录下建立了一个res101/voc_2007_trainval+voc_2012_trainval目录
# 并在其中新建了一个名称为default的软连接，连接至./data/voc数据集目录
```
测试Demo
```bash
# at repository root
GPU_ID=0
CUDA_VISIBLE_DEVICES=${GPU_ID} ./tools/demo.py
```
测试预训练的Resnet101模型：
```bash
GPU_ID=0
./experiments/scripts/test_faster_rcnn.sh $GPU_ID pascal_voc_0712 res101
```
训练自己的模型
```bash
# 预训练模型在data/imagenet_weights
# 使用VGG16模型训练时
mkdir -p data/imagenet_weights
cd data/imagenet_weights
wget -v http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xzvf vgg_16_2016_08_28.tar.gz
mv vgg_16.ckpt vgg16.ckpt
cd ../..
# 使用Resnet101模型训练时
mkdir -p data/imagenet_weights
cd data/imagenet_weights
wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
tar -xzvf resnet_v1_101_2016_08_28.tar.gz
mv resnet_v1_101.ckpt res101.ckpt
cd ../..
# 训练
./experiments/scripts/train_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
# NET in {vgg16, res50, res101, res152}
# DATASET in {pascal_voc, pascal_voc_0712, coco}
# Examples:
./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16
./experiments/scripts/train_faster_rcnn.sh 1 coco res101
# 确保在训练前删除预训练的软链接
# 用Tensorboard可视化
tensorboard --logdir=tensorboard/vgg16/voc_2007_trainval/ --port=7001 &
tensorboard --logdir=tensorboard/vgg16/coco_2014_train+coco_2014_valminusminival/ --port=7002 &
# 测试和评估
./experiments/scripts/test_faster_rcnn.sh [GPU_ID] [DATASET] [NET]
./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
./experiments/scripts/test_faster_rcnn.sh 1 coco res101
# 使用tools/reval.sh重新评估
# 默认的模型存放于 output/[NET]/[DATASET]/default/
```
## 目录结构
```bash
├── data //数据目录
│   ├── cache //训练集和测试集的proposals。
│   ├── cache.tgz  //程序首先从这读取。集合变化了一定先delete该目录下的文件
│   ├── coco                  //Python COCO API
│   ├── demo                  //保存几张图片(.jpg文件)，用于演示demo
│   ├── imagenet_weights      //存放预训练模型
│   ├── voc_2007_trainval     //下载的训练好的文件目录
│   ├── scripts               //一些实用脚本，用于获取数据
│   ├── VOCdevkit             //PASCAL VOC 2007数据集开发工具箱
│   ├── VOCdevkit2007 -> VOCdevkit   //软连接
│   └── wget-log              //下载模型的日志文件
================================================================
├── experiments
│   ├── cfgs                  //保存$NET.yml文件，针对具体网络的配置参数
│   ├── logs                  //保存每次训练和测试的日志
│   └── scripts               //保存三个.sh脚本，用于demo演示、测试和训练
│       ├── convert_vgg16.sh
│       ├── test_faster_rcnn.sh
│       └── train_faster_rcnn.sh
================================================================
├── lib    //主要的程序文件
│   ├── datasets   //读取数据的接口文件 
│   ├── layer_utils  //与anchor proposal相关
│   ├── Makefile
│   ├── model   //config配置文件   nms bbox test train_val等
│   ├── nets    //具体网络的程序文件(如mobilenet_v1，resnet_v1，vgg16)
│   ├── nms     //c和cuda的加速代码，生成共享库(.so)
│   ├── roi_data_layer  //RoI层
│   ├── setup.py  //用于构建Cython模块
│   └── utils  //一些辅助工具，计时、可视化 
================================================================
├── output  //保存训练模型和测试结果
│   ├── res101
│   └── vgg16
└── tools  // 训练、测试、演示等程序
=====================================================================
output目录
├── res101   //在faster-rcnn(res101)
│   ├── voc_2007_test    //测试结果，按类别保存的.pkl文件
│   │   └── default
│   ├── voc_2007_trainval   //训练的模型保存在该文件夹下
│   │   └── default
│   └── voc_2007_trainval+voc_2012_trainval  //训练好的模型(软链接)
└── vgg16
    ├── voc_2007_test
    ├── voc_2007_trainval
    └── voc_2007_trainval+voc_2012_trainval
```
