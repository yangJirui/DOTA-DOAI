# -*- coding: utf-8 -*-


from __future__ import absolute_import, print_function, division

import cv2

from data.io.image_preprocess import *
from libs.label_name_dict.label_dict import *
import numpy as np
import xml.etree.cElementTree as ET
import copy
import os

IMG_ROOT = '/home/yjr/DataSet/Dota_clip/val/images'
XML_ROOT = '/home/yjr/DataSet/Dota_clip/val/labeltxt'

def read_xml_gtbox_and_label(xml_path):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 9],
           and has [x1, y1, x2, y2, x3, y3, x4, y4, label] in a per row
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)

        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    label = NAME_LABEL_MAP[child_item.text]
                if child_item.tag == 'bndbox':
                    tmp_box = []
                    for node in child_item:
                        tmp_box.append(float(node.text))
                    assert label is not None, 'label is none, error'
                    tmp_box.append(label)
                    box_list.append(tmp_box)

    gtbox_label = np.array(box_list, dtype=np.int32)

    return img_width, img_height, gtbox_label


def show_roate_rect(img, gtbox_and_label):
    '''
    :param img:
    :param gtbox_and_label:
    :return:
    '''
    img = img
    for a_box in gtbox_and_label:
        # print(a_box)

        new_box = a_box[:-1]
        new_box = np.int0(new_box).reshape([4, 2])
        color = (np.random.randint(255),
                 np.random.randint(255),
                 np.random.randint(255))
        # print(type(color), color)
        cv2.drawContours(img, [new_box], -1, color, 2)
    # print (mask.dtype)
    return img

def test_rotate(img, gtbox_and_label, r_theta=45):

    img_copy = copy.deepcopy(img)
    img = show_roate_rect(img, gtbox_and_label)
    cv2.imwrite("raw_img.jpg", img)
    print("save_over1, gt_num:", len(gtbox_and_label))
    print(img_copy.dtype)
    r_img, gtbox_and_label = rotate_img_np(img_copy, gtbox_and_label, r_theta=r_theta)
    r_img = show_roate_rect(r_img, gtbox_and_label)
    print("after_rotate, shape: ", r_img.shape, len(gtbox_and_label))
    cv2.imwrite("rotate_img%d.jpg" % r_theta, r_img)

    return


if __name__ == "__main__":
    img_name = 'P0003_0000_0547'
    img = cv2.imread(os.path.join(IMG_ROOT, img_name+'.png'))
    img = np.array(img, dtype=np.float32)
    w, h, gtbox_and_label = read_xml_gtbox_and_label(os.path.join(XML_ROOT, img_name+'.xml'))
    print ("w, h:", (w, h))
    for r in [-30, -60, -90, 30, 60, 90]:
        test_rotate(img, gtbox_and_label, r)

