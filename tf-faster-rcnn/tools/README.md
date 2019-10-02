# tools

## demo.py
用于演示的程序
```python
# 定义了CLASSES = ('__background__', 'aeroplane',...)
# 定义NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),...}
# 定义DATASETS= {'pascal_voc': ('voc_2007_trainval',),...}
def vis_detections(im, class_name, dets, thresh=0.5):
    # 从dets中取出 bbox, score
    # 可视化

def demo(sess, net, image_name):
    # 读取图片image_name，cv2
    # 计时开始
    # 根据~/lib/model/test.py中的 im_detect(sess, net, im) 函数检测图片
    scores, boxes = im_detect(sess, net, im)
    # 计时结束
    # 设置score和nms的阈值，进行nms
    # 由下一个函数显示图片
    vis_detections(im, cls, dets, thresh=CONF_THRESH)

if __name__ == '__main__':
    # 获取命令行参数，--net和--dataset，读取对应的预训练模型.ckpt
    # 载入设置tf.ConfigProto，设置GPU: tfconfig.gpu_options.allow_growth
    # 初始化session
    # 初始化网络 net = vgg16() 或 net = resnetv1(num_layers=101)
    # 分别在 ~/lib/nets/vgg16.py 和 ~/lib/nets/resnet_v1.py
    # 然后具体参数传入，此功能定义于~/lib/nets/network.py 
    # 是 vgg16.py 和 resnet_v1.py 的基类
    net.create_architecture("TEST", 21, tag='default', anchor_scales=[8, 16, 32])
    # 创建 tf.train.Saver() 并恢复 restore from ckpt
    # 遍历测试图片
    demo(sess, net, im_name)
```
## trainval_net.py
用于加载训练参数、训练数据的程序，后续为`~/lib/model/train_val.py`
```bash
usage: trainval_net.py [-h] [--cfg CFG_FILE] [--weight WEIGHT]
                       [--imdb IMDB_NAME] [--imdbval IMDBVAL_NAME]
                       [--iters MAX_ITERS] [--tag TAG] [--net NET] [--set ...]

Train a Fast R-CNN network

optional arguments:
  -h, --help            show this help message and exit
  --cfg CFG_FILE        optional config file
  --weight WEIGHT       initialize with pretrained model weights
  --imdb IMDB_NAME      dataset to train on
  --imdbval IMDBVAL_NAME
                        dataset to validate on
  --iters MAX_ITERS     number of iterations to train
  --tag TAG             tag of the model
  --net NET             vgg16, res50, res101, res152, mobile
  --set ...             set config keys
```
程序
```python
def parse_args():
    # 获取命令行参数
def combined_roidb(imdb_names):
    # 结合几个数据集的函数
    def get_roidb(imdb_name):
        # 来自~/lib/datasets/factory.py，读取数据的接口
        imdb = get_imdb(imdb_name)
        # 来自~/lib/model/train_val.py的接口
        roidb = get_training_roidb(imdb)
        return roidb

if __name__ == '__main__':
  args = parse_args()  # 获取参数
  imdb, roidb = combined_roidb(args.imdb_name) # 加载训练数据
  # 设置输出目录
  _, valroidb = combined_roidb(args.imdbval_name)# 加载测试数据
  # 是否翻转
  #构建网络
  train_net(net,imdb,roidb,valroidb,
            output_dir,tb_dir,
            pretrained_model=args.weight,
            max_iters=args.max_iters)
    

```
程序输出：
```bash
Called with args:
Namespace(cfg_file=None, imdb_name='voc_2007_trainval', imdbval_name='voc_2007_test', max_iters=70000, net='vgg16', set_cfgs=None, tag=None, weight='/home/seucar/Desktop/tf-faster-rcnn/data/imagenet_weights/vgg16.ckpt')
Using config:
{'ANCHOR_RATIOS': [0.5, 1, 2],
 'ANCHOR_SCALES': [8, 16, 32],
 'DATA_DIR': '/home/seucar/Desktop/tf-faster-rcnn/data',
 'EXP_DIR': 'default',
 'MATLAB': 'matlab',
 'MOBILENET': {'DEPTH_MULTIPLIER': 1.0,
               'FIXED_LAYERS': 5,
               'REGU_DEPTH': False,
               'WEIGHT_DECAY': 4e-05},
 'PIXEL_MEANS': array([[[102.9801, 115.9465, 122.7717]]]),
 'POOLING_MODE': 'crop',
 'POOLING_SIZE': 7,
 'RESNET': {'FIXED_BLOCKS': 1, 'MAX_POOL': False},
 'RNG_SEED': 3,
 'ROOT_DIR': '/home/seucar/Desktop/tf-faster-rcnn',
 'RPN_CHANNELS': 512,
 'TEST': {'BBOX_REG': True,
          'HAS_RPN': False,
          'MAX_SIZE': 1000,
          'MODE': 'nms',
          'NMS': 0.3,
          'PROPOSAL_METHOD': 'gt',
          'RPN_NMS_THRESH': 0.7,
          'RPN_POST_NMS_TOP_N': 300,
          'RPN_PRE_NMS_TOP_N': 6000,
          'RPN_TOP_N': 5000,
          'SCALES': [600],
          'SVM': False},
 'TRAIN': {'ASPECT_GROUPING': False,
           'BATCH_SIZE': 128,
           'BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
           'BBOX_NORMALIZE_MEANS': [0.0, 0.0, 0.0, 0.0],
           'BBOX_NORMALIZE_STDS': [0.1, 0.1, 0.2, 0.2],
           'BBOX_NORMALIZE_TARGETS': True,
           'BBOX_NORMALIZE_TARGETS_PRECOMPUTED': True,
           'BBOX_REG': True,
           'BBOX_THRESH': 0.5,
           'BG_THRESH_HI': 0.5,
           'BG_THRESH_LO': 0.1,
           'BIAS_DECAY': False,
           'DISPLAY': 10,
           'DOUBLE_BIAS': True,
           'FG_FRACTION': 0.25,
           'FG_THRESH': 0.5,
           'GAMMA': 0.1,
           'HAS_RPN': True,
           'IMS_PER_BATCH': 1,
           'LEARNING_RATE': 0.001,
           'MAX_SIZE': 1000,
           'MOMENTUM': 0.9,
           'PROPOSAL_METHOD': 'gt',
           'RPN_BATCHSIZE': 256,
           'RPN_BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
           'RPN_CLOBBER_POSITIVES': False,
           'RPN_FG_FRACTION': 0.5,
           'RPN_NEGATIVE_OVERLAP': 0.3,
           'RPN_NMS_THRESH': 0.7,
           'RPN_POSITIVE_OVERLAP': 0.7,
           'RPN_POSITIVE_WEIGHT': -1.0,
           'RPN_POST_NMS_TOP_N': 2000,
           'RPN_PRE_NMS_TOP_N': 12000,
           'SCALES': [600],
           'SNAPSHOT_ITERS': 5000,
           'SNAPSHOT_KEPT': 3,
           'SNAPSHOT_PREFIX': 'res101_faster_rcnn',
           'STEPSIZE': [30000],
           'SUMMARY_INTERVAL': 180,
           'TRUNCATED': False,
           'USE_ALL_GT': True,
           'USE_FLIPPED': True,
           'USE_GT': False,
           'WEIGHT_DECAY': 0.0001},
 'USE_E2E_TF': True,
 'USE_GPU_NMS': True}
Loaded dataset `voc_2007_trainval` for training
Set proposal method: gt
Appending horizontally-flipped training examples...
wrote gt roidb to /home/seucar/Desktop/tf-faster-rcnn/data/cache/voc_2007_trainval_gt_roidb.pkl
done
Preparing training data...
done
10022 roidb entries
Output will be saved to `/home/seucar/Desktop/tf-faster-rcnn/output/default/voc_2007_trainval/default`
TensorFlow summaries will be saved to `/home/seucar/Desktop/tf-faster-rcnn/tensorboard/default/voc_2007_trainval/default`
Loaded dataset `voc_2007_test` for training
Set proposal method: gt
Preparing training data...
wrote gt roidb to /home/seucar/Desktop/tf-faster-rcnn/data/cache/voc_2007_test_gt_roidb.pkl
done
4952 validation roidb entries
Filtered 0 roidb entries: 10022 -> 10022
Filtered 0 roidb entries: 4952 -> 4952
Loading initial model weights from /home/seucar/Desktop/tf-faster-rcnn/data/imagenet_weights/vgg16.ckpt
Variables restored: vgg_16/conv1/conv1_1/biases:0
Variables restored: vgg_16/conv1/conv1_2/weights:0
Variables restored: vgg_16/conv1/conv1_2/biases:0
Variables restored: vgg_16/conv2/conv2_1/weights:0
Variables restored: vgg_16/conv2/conv2_1/biases:0
Variables restored: vgg_16/conv2/conv2_2/weights:0
Variables restored: vgg_16/conv2/conv2_2/biases:0
Variables restored: vgg_16/conv3/conv3_1/weights:0
Variables restored: vgg_16/conv3/conv3_1/biases:0
Variables restored: vgg_16/conv3/conv3_2/weights:0
Variables restored: vgg_16/conv3/conv3_2/biases:0
Variables restored: vgg_16/conv3/conv3_3/weights:0
Variables restored: vgg_16/conv3/conv3_3/biases:0
Variables restored: vgg_16/conv4/conv4_1/weights:0
Variables restored: vgg_16/conv4/conv4_1/biases:0
Variables restored: vgg_16/conv4/conv4_2/weights:0
Variables restored: vgg_16/conv4/conv4_2/biases:0
Variables restored: vgg_16/conv4/conv4_3/weights:0
Variables restored: vgg_16/conv4/conv4_3/biases:0
Variables restored: vgg_16/conv5/conv5_1/weights:0
Variables restored: vgg_16/conv5/conv5_1/biases:0
Variables restored: vgg_16/conv5/conv5_2/weights:0
Variables restored: vgg_16/conv5/conv5_2/biases:0
Variables restored: vgg_16/conv5/conv5_3/weights:0
Variables restored: vgg_16/conv5/conv5_3/biases:0
Variables restored: vgg_16/fc6/biases:0
Variables restored: vgg_16/fc7/biases:0
Loaded.
Fix VGG16 layers..
Fixed.
iter: 10 / 70000, total loss: 1.617437
 >>> rpn_loss_cls: 0.442162
 >>> rpn_loss_box: 0.002378
 >>> loss_cls: 0.752117
 >>> loss_box: 0.288937
 >>> lr: 0.001000
speed: 3.366s / iter
iter: 20 / 70000, total loss: 2.135787
 >>> rpn_loss_cls: 0.379773
 >>> rpn_loss_box: 0.169955
 >>> loss_cls: 0.962757
 >>> loss_box: 0.491440
 >>> lr: 0.001000
speed: 3.211s / iter
...
```






















```

```