# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
from help_utils.vis_utils_for_test import draw_box_cv
import matplotlib.pyplot as plt
import cv2
from help_utils import vis_utils_for_test


def make_rotate_anchors(base_anchor_size, anchor_scales, anchor_ratios, anchor_angles, fet_h, fet_w, stride):

    base_anchor = np.array([0, 0, base_anchor_size, base_anchor_size], np.float32)
    ws, hs, angles = enum_ratios_and_thetas(enum_scales(base_anchor, anchor_scales),
                                            anchor_ratios, anchor_angles)
    x_centers = np.arange(0, fet_w, dtype=np.float32) * stride
    y_centers = np.arange(0, fet_h, dtype=np.float32) * stride

    x_centers, y_centers = np.meshgrid(x_centers, y_centers)

    angles, _ = np.meshgrid(angles, x_centers)

    ws, x_centers = np.meshgrid(ws, x_centers)
    hs, y_centers = np.meshgrid(hs, y_centers)

    anchor_centers = np.stack([x_centers, y_centers], axis=2)
    anchor_centers = np.reshape(anchor_centers, [-1, 2])

    box_parameters = np.stack([ws, hs, angles], axis=2)
    box_parameters = np.reshape(box_parameters, [-1, 3])
    anchors = np.concatenate((anchor_centers, box_parameters), axis=1)

    return anchors


def enum_scales(base_anchor, anchor_scales):

    anchor_scales = np.array(anchor_scales, dtype=np.float32).reshape((-1, 1))
    anchors = base_anchor * anchor_scales

    return anchors


def enum_ratios_and_thetas(anchors, anchor_ratios, anchor_angles):
    '''
    ratio = h /w
    :param anchors:
    :param anchor_ratios:
    :return:
    '''
    ws = anchors[:, 2]  # for base anchor: w == h
    hs = anchors[:, 3]
    anchor_angles = np.array(anchor_angles)  # tf.constant(anchor_angles, tf.float32)
    sqrt_ratios = np.sqrt(np.array(anchor_ratios, dtype=np.float32))

    ws = np.reshape(ws / sqrt_ratios[:, np.newaxis], [-1])
    hs = np.reshape(hs * sqrt_ratios[:, np.newaxis], [-1])

    ws, _ = np.meshgrid(ws, anchor_angles)
    hs, anchor_angles = np.meshgrid(hs, anchor_angles)

    anchor_angles = np.reshape(anchor_angles, [-1, 1])
    ws = np.reshape(ws, [-1, 1])
    hs = np.reshape(hs, [-1, 1])

    return hs, ws, anchor_angles




def test_convexHull(img, boxes):
    for i, box in enumerate(boxes):
        x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]
        rect = ((x_c, y_c), (w, h), theta)
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)

        color = (128, 128, 0)
        cv2.drawContours(img, [rect], -1, color, 3)

        index = np.arange(len(rect))
        np.random.shuffle(index)
        rect = rect[index]
        rect = np.asarray(rect, dtype=np.float32)
        point_list = vis_utils_for_test.order_points(rect)
        # point_list = vis_utils_for_test.sort_points(rect)
        point_list = np.asarray(point_list, dtype=np.int)

        # point_list = cv2.convexHull(points=rect, returnPoints=True, clockwise=True)
        # point_list = np.squeeze(point_list, axis=1)
        p_colors = [(255, 255, 255), (0, 255, 0), (0, 0, 255), (128, 0, 0)]
        for p, p_color in zip(point_list, p_colors):
            p = p.tolist()
            p = tuple(p)
            cv2.circle(img, center=p, radius=10, color=p_color, thickness=-1)

    return img
def test_a_location(anchor_angles, anchor_ratios, anchor_scales=[1.0]):
    base_anchor = np.array([0, 0, 100, 100], np.float32)
    ws, hs, angles = enum_ratios_and_thetas(enum_scales(base_anchor, anchor_scales),
                                            anchor_ratios, anchor_angles)

    print (ws.shape, hs.shape, angles.shape)
    xc, yc = 110, 110
    xc = np.ones_like(ws) * xc
    yc = np.ones_like(hs) * yc

    xc = np.arange(130, 660, 100).reshape((-1, 1))
    yc = xc
    a_location_anchor = np.concatenate([xc, yc, ws, hs, angles], axis=1)

    img = np.zeros(shape=(800, 800, 3), dtype=np.uint8)

    # img = draw_box_cv(img, a_location_anchor)
    img = test_convexHull(img, a_location_anchor)
    plt.imshow(img)

    plt.show()


if __name__ == '__main__':

    test_a_location(anchor_ratios=[1/2.],
                    anchor_angles=[-90, -75, -60, -45, -30, -15])
    # base_anchor_size = 256
    # anchor_scales = [1.]
    # anchor_ratios = [0.5, 2.0, 1/3, 3, 1/5, 5, 1/8, 8]
    # anchor_angless = [-90, -75, -60, -45, -30, -15]
    # base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], tf.float32)
    # tmp1 = enum_ratios_and_thetas(enum_scales(base_anchor, anchor_scales), anchor_ratios, anchor_angless)
    # anchors = make_anchors(32,
    #                [1.], anchor_ratios, [-90, -75, -60, -45, -30, -15],
    #                featuremap_height=600 // 16 * 2,
    #                featuremap_width=1000 // 16 * 2,
    #                stride=8)
    #
    # img = tf.ones([600, 1000, 3])
    # img = tf.expand_dims(img, axis=0)
