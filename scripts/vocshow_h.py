import xml.etree.ElementTree as ET
import os
from os import getcwd

import cv2

from tools import  *

classes = ["1","2","3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]


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


classesname = ["SQU_WHITE", "SQU_PINK", "SQU_BLUE", "SQU_GREEN", "CIR_WHITE", "CANPAN", "DRINK1", "SOUP", "MILK",
                "YANGLEDUO", "MILK", "MILK"]
colors = [(127, 127, 127), (127, 127, 255),
                       (255, 0, 0), (127, 255, 127),
                       (0, 0, 0), (255, 255, 0), (0,0,255),(0,255,0),(255,255,0),(0,255,255),(255,0,0),(255,0,0)]
def drawPred(frame, classId, conf, left, top, right, bottom, color):
    # Draw a bounding box.
    cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), color, thickness=2)

    label = '%.2f' % conf
    label = '%s' % (classesname[classId])

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
    cv2.putText(frame, label, (int(left), int(top - 10)), cv2.FONT_HERSHEY_TRIPLEX, 1, color, thickness=1)
    return frame


def convert_annotation(xmlp, imgp, savedir):
    in_file = open(xmlp, encoding="utf-8")
    tree = ET.parse(in_file)
    root = tree.getroot()

    img = cv2.imread(imgp)
    h,w = img.shape[:2]

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))

        img = drawPred(img, cls_id, 1, b[0], b[2], b[1], b[3], colors[cls_id])
    # cv2.imshow("tmp", img)
    # cv2.waitKey(0)
    savep = "{}/{}".format(savedir, os.path.basename(imgp))
    cv2.imwrite(savep, img)
    in_file.close()


import glob
imgdir = r"D:\data\work\canpan\datasets\train\20221023real\JPEGImages"
savedir = r"D:\data\work\canpan\datasets\train\20221023real\1323show"
if not os.path.exists(savedir):
    os.mkdir(savedir)
imgplist = glob.glob("{}/*.jpg".format(imgdir))
for imgp in imgplist:
    if imgp.find("_90") >0 or imgp.find("_180") >0 or imgp.find("_270") >0 :
        continue
    xmlp = (imgp[:-3]+'xml').replace('JPEGImages', 'Annotations')
    print(imgp)
    print(xmlp)
    convert_annotation(xmlp, imgp, savedir)

