# -*- coding: utf-8 -*-


from __future__ import absolute_import, print_function, division

import cv2

# from data.io.image_preprocess import *
from libs.label_name_dict.label_dict import *
from libs.box_utils import mask_utils

import numpy as np
import xml.etree.cElementTree as ET
import copy
import os

IMG_ROOT = '/home/omnisky/DataSets/Dota_clip/trainval800/images'
XML_ROOT = '/home/omnisky/DataSets/Dota_clip/trainval800/labeltxt'

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

def rotate_img_np(img, gtboxes_and_label, r_theta):
    h, w, c = img.shape
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, r_theta, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    nW, nH = int(h*sin + w*cos), int(h*cos + w*sin)  # new W and new H
    M[0, 2] += (nW/2) - center[0]
    M[1, 2] += (nH/2) - center[1]
    rotated_img = cv2.warpAffine(img, M, (nW, nH))
    # -------

    new_points_list = []
    obj_num = len(gtboxes_and_label)
    for st in range(0, 7, 2):
        points = gtboxes_and_label[:, st:st+2]
        expand_points = np.concatenate((points, np.ones(shape=(obj_num, 1))), axis=1)
        new_points = np.dot(M, expand_points.T)
        new_points = new_points.T
        new_points_list.append(new_points)
    gtboxes = np.concatenate(new_points_list, axis=1)
    gtboxes_and_label = np.concatenate((gtboxes, gtboxes_and_label[:, -1].reshape(-1, 1)), axis=1)

    # x1, y1, x2, y2, x3, y3, x4, y4 = np.split(gtboxes, 8, 1)
    #
    # xc = 0.25*(x1+x2+x3+x4)
    # yc = 0.25*(y1+y2+y3+y4)
    #
    # valid = (xc>0) & (yc>0) & (xc<w) & (yc<h)
    # valid = valid.reshape((-1, ))
    # # print (valid)
    # if np.sum(valid) != 0:
    #     gtboxes_and_label = gtboxes_and_label[valid]
    # gtboxes_and_label = np.asarray(gtboxes_and_label, dtype=np.int32)

    return rotated_img, gtboxes_and_label

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


def test_mask(img, gtbox_and_label):

    h, w, c= img.shape
    img = show_roate_rect(img, gtbox_and_label)

    img =



if __name__ == "__main__":
    # img_name = 'P0003_0000_0547'
    img_name = 'P0003_0000_0347'
    img = cv2.imread(os.path.join(IMG_ROOT, img_name+'.png'))
    img = np.array(img, dtype=np.float32)
    w, h, gtbox_and_label = read_xml_gtbox_and_label(os.path.join(XML_ROOT, img_name+'.xml'))
    print ("w, h:", (w, h))
    for r in [-30, -60, -90, 30, 60, 90]:
        test_rotate(img, gtbox_and_label, r)

