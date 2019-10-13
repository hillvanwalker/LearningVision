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

## test_net.py






















