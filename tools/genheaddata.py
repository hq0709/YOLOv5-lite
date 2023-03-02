import cv2
import xml.etree.ElementTree as ET
import numpy as np
from writeXml import save_annotations_and_imgs
import os

classes = ["1"]  # 我们只检测person这个类别
def get_annotation(xmlp):
    # in_file = open(xmlp, encoding='gbk')
    in_file = open(xmlp, encoding="utf-8")
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')

    ret = [] # [[xl,yl,xr,yr]]
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymax').text))
        ret.append(b)

    return ret

def gendata(imgp, winannos, headannos, outputdir):
    img = cv2.imread(imgp)
    imgh,imgw = img.shape[:2]
    expsc = 1.1 # 外扩1.1倍

    for count, winanno in enumerate(winannos):
        xmin, ymin, xmax, ymax = winanno
        xc, yc, w, h = (xmin+xmax)*0.5, (ymin+ymax)*0.5, (xmax-xmin+1), (ymax-ymin+1)
        xmin_n, ymin_n, xmax_n, ymax_n = xc - w*expsc*0.5, yc-h*expsc*0.5, xc + w*expsc*0.5, yc+h*expsc*0.5
        xmin_n = max(0, int(xmin_n))
        ymin_n = max(0, int(ymin_n))
        xmax_n = min(imgw, int(xmax_n))
        ymax_n = min(imgh, int(ymax_n))

        imgcrop = img[ymin_n:ymax_n, xmin_n:xmax_n]
        headannos_new = []
        for headanno in headannos:
            xmin_head, ymin_head, xmax_head, ymax_head = headanno
            # 如果人头中点在车窗中，则更新人头坐标
            if xmin < (xmin_head+xmax_head)*0.5 < xmax and ymin < (ymin_head+ymax_head)*0.5 < ymax:
                annotmp = [1,int(xmin_head-xmin_n), int(ymin_head-ymin_n),int(xmax_head-xmin_n), int(ymax_head-ymin_n)]
                headannos_new.append(annotmp)
        #         imgcrop = cv2.rectangle(imgcrop, (annotmp[1], annotmp[2]), (annotmp[3], annotmp[4]), (0,0,255))
        # cv2.imshow('tmp', imgcrop)
        # cv2.waitKey(0)
        saveimgp = os.path.join(outputdir, os.path.basename(imgp)[:-4]+"_{}.jpg".format(count))
        cv2.imwrite(saveimgp, imgcrop)
        save_annotations_and_imgs(saveimgp, headannos_new, imgcrop.shape)

winlabelpath = r"D:\data\work\jgtd\src\batch20220429\1\winlabel.txt"
outputdir = r'D:\data\work\jgtd\src\head_20220510\JPEGImages'
winlabels = []
with open(winlabelpath,'r') as f:
    winlabels = [i.strip() for i in f.readlines()]

for line in winlabels:
    lds = line.split(' ')
    imgp = lds[0]
    annop = imgp.replace('JPEGImages', 'Annotations').replace('jpg', 'xml')
    if not os.path.exists(annop):
        continue
    headannos = get_annotation(annop)
    winannos = np.array([float(i) for i in lds[1:]]).reshape((-1,4))

    gendata(imgp, winannos, headannos, outputdir)

    # print(winannos)
    #
    # imgshow = cv2.imread(imgp)
    # for winann in winannos:
    #     imgshow= cv2.rectangle(imgshow, (int(winann[0]), int(winann[1])), (int(winann[2]), int(winann[3])), (0,0,255))
    # for headann in headannos:
    #     imgshow = cv2.rectangle(imgshow, (int(headann[0]), int(headann[1])), (int(headann[2]), int(headann[3])),
    #                             (0, 255, 255))
    # cv2.imshow('tmp', imgshow)
    # cv2.waitKey(0)