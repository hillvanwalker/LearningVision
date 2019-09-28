import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
# lib_path = osp.join(this_dir, '..', 'lib')
# add_path(lib_path)
add_path('/home/seucar/Desktop/tf-faster-rcnn/lib')

# coco_path = osp.join(this_dir, '..', 'data', 'coco', 'PythonAPI')
# add_path(coco_path)
add_path('/home/seucar/Desktop/tf-faster-rcnn/data/coco/PythonAPI')