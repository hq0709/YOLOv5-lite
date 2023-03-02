import xml.etree.ElementTree as ET
import os
from os import getcwd

import cv2

from tools import  *

classes = ["1","2","3", "4", "5", "666", "7", "8", "9", "10", "11", "12", "133"]
# classes = ["1","2","3", "4", "5", "6"]
# classes = ["1"] # 只检测人头1类别
# classes = ["person"] # 只检测人头1类别


def convert(size, box):#对图片进行归一化处理
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(xmlp, labelp, imgp):
    # in_file = open(xmlp, encoding='gbk')
    in_file = open(xmlp, encoding="utf-8")
    out_file = open(labelp, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()

    # size = root.find('size')
    # w = int(size.find('width').text)
    # h = int(size.find('height').text)

    img = cv2.imread(imgp)
    h,w = img.shape[:2]

    for obj in root.iter('object'):
        tmp = obj.find('difficult')
        if tmp is None:
            tmp = obj.find('Difficult')
        difficult = tmp.text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    in_file.close()
    out_file.close()


import glob
imgdir = r"D:\data\work\canpan\datasets\train\20230216real\JPEGImages"
imgplist = glob.glob("{}/*.jpg".format(imgdir))
for imgp in imgplist:
    labelp = imgp[:-3]+'txt'
    xmlp = (imgp[:-3]+'xml').replace('JPEGImages', 'Annotations')
    print(imgp)
    print(labelp)
    print(xmlp)
    convert_annotation(xmlp, labelp, imgp)

