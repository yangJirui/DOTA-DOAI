# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
from libs.label_name_dict.label_dict import LABEl_NAME_MAP

from libs.configs import cfgs

from libs.box_utils import draw_box_in_img


def only_draw_boxes(img_batch, boxes):

    boxes = tf.stop_gradient(boxes)
    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    labels = tf.ones(shape=(tf.shape(boxes)[0], ), dtype=tf.int32) * draw_box_in_img.ONLY_DRAW_BOXES
    scores = tf.zeros_like(labels, dtype=tf.float32)
    img_tensor_with_boxes = tf.py_func(draw_box_in_img.draw_boxes_with_label_and_scores,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=tf.uint8)
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))  # [batch_size, h, w, c]

    return img_tensor_with_boxes


def draw_boxes_with_scores(img_batch, boxes, scores):

    boxes = tf.stop_gradient(boxes)
    scores = tf.stop_gradient(scores)

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    labels = tf.ones(shape=(tf.shape(boxes)[0],), dtype=tf.int32) * draw_box_in_img.ONLY_DRAW_BOXES_WITH_SCORES
    img_tensor_with_boxes = tf.py_func(draw_box_in_img.draw_boxes_with_label_and_scores,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes


def draw_boxes_with_categories(img_batch, boxes, labels):
    boxes = tf.stop_gradient(boxes)

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    scores = tf.ones(shape=(tf.shape(boxes)[0],), dtype=tf.float32)
    img_tensor_with_boxes = tf.py_func(draw_box_in_img.draw_boxes_with_label_and_scores,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes


def draw_boxes_with_categories_and_scores(img_batch, boxes, labels, scores):
    boxes = tf.stop_gradient(boxes)
    scores = tf.stop_gradient(scores)

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor_with_boxes = tf.py_func(draw_box_in_img.draw_boxes_with_label_and_scores,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes


def draw_box_with_color_rotate(img_batch, boxes, text):

    def draw_box_cv(img, boxes, text):
        if cfgs.MXNET_NORM:
            img = img * np.array(cfgs.MXNET_STD)
            img = img + np.array(cfgs.MXNET_MEAN)
            img = img * 255.0
        else:
            img = img + np.array(cfgs.PIXEL_MEAN)
        boxes = boxes.astype(np.int64)
        img = np.array(img * 255 / np.max(img), np.uint8)
        for box in boxes:
            x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]
            rect = ((x_c, y_c), (w, h), theta)
            rect = cv2.boxPoints(rect)
            rect = np.int0(rect)
            color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            cv2.drawContours(img, [rect], -1, color, 3)

        text = str(text)
        cv2.putText(img,
                    text=text,
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))
        # img = img[:, :, ::-1]
        return img

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, text],
                                       Tout=[tf.uint8])

    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))

    return img_tensor_with_boxes


def draw_boxes_with_categories_and_scores_rotate(img_batch, boxes, labels, scores):

    def draw_box_cv(img, boxes, labels, scores):
        if cfgs.MXNET_NORM:
            img = img * np.array(cfgs.MXNET_STD)
            img = img + np.array(cfgs.MXNET_MEAN)
            img = img * 255.0
        else:
            img = img + np.array(cfgs.PIXEL_MEAN)
        boxes = boxes.astype(np.int64)
        labels = labels.astype(np.int32)
        img = np.array(img*255/np.max(img), np.uint8)

        num_of_object = 0
        for i, box in enumerate(boxes):

            x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]
            label = labels[i]
            score = scores[i]
            if label != 0:
                num_of_object += 1

                rect = ((x_c, y_c), (w, h), theta)
                rect = cv2.boxPoints(rect)
                rect = np.int0(rect)
                color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
                cv2.drawContours(img, [rect], -1, color, 3)

                cv2.rectangle(img,
                              pt1=(x_c, y_c),
                              pt2=(x_c+120, y_c+15),
                              color=color,
                              thickness=-1)
                category = LABEl_NAME_MAP[label]
                cv2.putText(img,
                            text=category+": "+str(score),
                            org=(x_c, y_c+10),
                            fontFace=1,
                            fontScale=1,
                            thickness=2,
                            color=(color[1], color[2], color[0]))
        cv2.putText(img,
                    text=str(num_of_object),
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))
        # img = img[:, :, ::-1]
        return img

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes



if __name__ == "__main__":
    print (1)

