from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import random

headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''


# 如果目录不存在则创建它，存在则删除后创建
def mkr(path):
    if os.path.exists(path):
        # 删除后创建
        shutil.rmtree(path)
        # os.mkdir(path)
        # os.mkdir创建单层目录；os.makedirs创建多层目录
        os.makedirs(path)
    else:
        os.makedirs(path)


# 通过coco数据集的id，得到它的类别名
def id2name(coco):
    classes = dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']
    return classes


# 写xml文件
def write_xml(anno_path, head, objs, tail):
    f = open(anno_path, "w")
    f.write(head)
    for obj in objs:
        f.write(objstr % (obj[0], obj[1], obj[2], obj[3], obj[4]))
    f.write(tail)


#
def save_annotations_and_imgs(filename, objs, imgshape):
    # eg:COCO_train2014_000000196610.jpg-->COCO_train2014_000000196610.xml
    anno_path = filename.replace("JPEGImages", "Annotations")[:-3] + 'xml'
    img_path=filename
    print(img_path)

    head = headstr % (filename, imgshape[1], imgshape[0], imgshape[2])
    tail = tailstr
    write_xml(anno_path, head, objs, tail)
