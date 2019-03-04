# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys

sys.path.append("../")
import cv2
import numpy as np
from timeit import default_timer as timer
import tensorflow as tf

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.networks import build_whole_network
from help_utils.tools import *
from libs.box_utils import draw_box_in_img
# from libs.box_utils import coordinate_convert
from libs.label_name_dict.label_dict import LABEl_NAME_MAP, NAME_LABEL_MAP
from help_utils import tools
# from libs.box_utils import nms
from libs.box_utils.cython_utils.cython_nms import nms, soft_nms
from libs.configs import cfgs
from PIL import Image, ImageDraw, ImageFont
FONT = ImageFont.load_default()

def get_file_paths_recursive(folder=None, file_ext=None):
    """ Get the absolute path of all files in given folder recursively
    :param folder:
    :param file_ext:
    :return:
    """
    file_list = []
    if folder is None:
        return file_list
    file_list = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(file_ext)]

    return file_list

def draw_for_paper(img_array, boxes, labels, scores):

    if cfgs.MXNET_NORM:
        img_array = img_array * np.array(cfgs.MXNET_STD)
        img_array = img_array + np.array(cfgs.MXNET_MEAN)
        img_array = img_array * 255.0
    else:
        img_array = img_array + np.array(cfgs.PIXEL_MEAN)
    img_array.astype(np.float32)
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)
    img_array = np.array(img_array * 255 / np.max(img_array), dtype=np.uint8)

    img_obj = Image.fromarray(img_array)
    raw_img_obj = img_obj.copy()

    draw_obj = ImageDraw.Draw(img_obj)
    for box, a_label, a_score in zip(boxes, labels, scores):
        draw_box_in_img.draw_a_rectangel_in_img(draw_obj, box,
                                                color='Red',
                                                width=3)
        # color = 'White'
        # x, y = box[0], box[1]
        # draw_obj.rectangle(xy=[x, y, x + 60, y + 10],
        #                    fill=color)
        # draw_obj.text(xy=(x, y),
        #               text="obj:" + str(round(a_score, 2)),
        #               fill='black',
        #               font=FONT)
    out_img_obj = Image.blend(raw_img_obj, img_obj, alpha=1.0)

    return np.array(out_img_obj)




def inference(det_net, file_paths, des_folder, h_len, w_len, h_overlap, w_overlap, save_res=False):
    TMP_FILE = './tmp_%s.txt' % cfgs.VERSION

    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
    img_batch = tf.cast(img_plac, tf.float32)
    if cfgs.MXNET_NORM:
        print("USe Mxnet Norm...\n")
        img_batch = img_batch / 255.0
        img_batch = img_batch - tf.constant(cfgs.MXNET_MEAN)
        img_batch = img_batch / tf.constant(cfgs.MXNET_STD)
    else:
        img_batch = img_batch - tf.constant([[cfgs.PIXEL_MEAN]])  # sub pixel mean at last
        if cfgs.NET_NAME.endswith(('b', 'd')):
            print("Note: Use Mxnet ResNet, But Do Not Norm Img like MxNet....")
            print('\n')
    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN[0],
                                                     is_resize=False)

    det_boxes_h, det_scores_h, det_category_h = det_net.build_whole_detection_network(input_img_batch=img_batch,
                                                                                      gtboxes_batch=None,
                                                                                      gtboxes_r_batch=None)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = det_net.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    print('yjr filter box with score 1e-4 ***************')
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        if not os.path.exists(TMP_FILE):
            fw = open(TMP_FILE, 'w')
            fw.close()

        fr = open(TMP_FILE, 'r')
        pass_img = fr.readlines()
        fr.close()

        for count, img_path in enumerate(file_paths):
            fw = open(TMP_FILE, 'a+')
            if img_path + '\n' in pass_img:
                continue
            start = timer()
            img = cv2.imread(img_path)

            box_res = []
            label_res = []
            score_res = []

            imgH = img.shape[0]
            imgW = img.shape[1]
            ori_H = imgH
            ori_W = imgW
            print("  ori_h, ori_w: ", imgH, imgW)
            if imgH < h_len:
                temp = np.zeros([h_len, imgW, 3], np.float32)
                temp[0:imgH, :, :] = img
                img = temp
                imgH = h_len

            if imgW < w_len:
                temp = np.zeros([imgH, w_len, 3], np.float32)
                temp[:, 0:imgW, :] = img
                img = temp
                imgW = w_len

            for hh in range(0, imgH, h_len - h_overlap):
                if imgH - hh - 1 < h_len:
                    hh_ = imgH - h_len
                else:
                    hh_ = hh
                for ww in range(0, imgW, w_len - w_overlap):
                    if imgW - ww - 1 < w_len:
                        ww_ = imgW - w_len
                    else:
                        ww_ = ww
                    src_img = img[hh_:(hh_ + h_len), ww_:(ww_ + w_len), :]

                    for short_size in cfgs.IMG_SHORT_SIDE_LEN:
                        max_len = 1200

                        if h_len < w_len:
                            new_h, new_w = short_size,  min(int(short_size*float(w_len)/h_len), max_len)
                        else:
                            new_h, new_w = min(int(short_size*float(h_len)/w_len), max_len), short_size

                        img_resize = cv2.resize(src_img, (new_h, new_w))

                        det_boxes_h_, det_scores_h_, det_category_h_ = \
                            sess.run(
                                [det_boxes_h, det_scores_h, det_category_h],
                                feed_dict={img_plac: img_resize[:, :, ::-1]}
                            )
                        # -=------

                        valid = det_scores_h_ > 1e-4
                        det_boxes_h_ = det_boxes_h_[valid]
                        det_scores_h_ = det_scores_h_[valid]
                        det_category_h_ = det_category_h_[valid]
                        # ---------
                        det_boxes_h_[:, 0] = det_boxes_h_[:, 0] * w_len / new_w
                        det_boxes_h_[:, 1] = det_boxes_h_[:, 1] * h_len / new_h
                        det_boxes_h_[:, 2] = det_boxes_h_[:, 2] * w_len / new_w
                        det_boxes_h_[:, 3] = det_boxes_h_[:, 3] * h_len / new_h

                        if len(det_boxes_h_) > 0:
                            for ii in range(len(det_boxes_h_)):
                                box = det_boxes_h_[ii]
                                box[0] = box[0] + ww_
                                box[1] = box[1] + hh_
                                box[2] = box[2] + ww_
                                box[3] = box[3] + hh_
                                box_res.append(box)
                                label_res.append(det_category_h_[ii])
                                score_res.append(det_scores_h_[ii])

            box_res = np.array(box_res)
            label_res = np.array(label_res)
            score_res = np.array(score_res)

            box_res_, label_res_, score_res_ = [], [], []

            h_threshold = {'roundabout': 0.5, 'tennis-court': 0.5, 'swimming-pool': 0.5, 'storage-tank': 0.5,
                           'soccer-ball-field': 0.5, 'small-vehicle': 0.5, 'ship': 0.5, 'plane': 0.5,
                           'large-vehicle': 0.5, 'helicopter': 0.5, 'harbor': 0.5, 'ground-track-field': 0.5,
                           'bridge': 0.5, 'basketball-court': 0.5, 'baseball-diamond': 0.5}

            for sub_class in range(1, cfgs.CLASS_NUM + 1):
                index = np.where(label_res == sub_class)[0]
                if len(index) == 0:
                    continue
                tmp_boxes_h = box_res[index]
                tmp_label_h = label_res[index]
                tmp_score_h = score_res[index]

                tmp_boxes_h = np.array(tmp_boxes_h)
                tmp = np.zeros([tmp_boxes_h.shape[0], tmp_boxes_h.shape[1] + 1])
                tmp[:, 0:-1] = tmp_boxes_h
                tmp[:, -1] = np.array(tmp_score_h)

                # inx = nms.py_cpu_nms(dets=np.array(tmp, np.float32),
                #                      thresh=h_threshold[LABEL_NAME_MAP[sub_class]],
                #                      max_output_size=500)

                if cfgs.SOFT_NMS:
                    inx = soft_nms(np.array(tmp, np.float32), 0.5, Nt=h_threshold[LABEl_NAME_MAP[sub_class]],
                                   threshold=0.001, method=2)  # 2 means Gaussian
                else:
                    inx = nms(np.array(tmp, np.float32),
                              h_threshold[LABEl_NAME_MAP[sub_class]])

                inx = inx[:500]  # max_outpus is 500

                box_res_.extend(np.array(tmp_boxes_h)[inx])
                score_res_.extend(np.array(tmp_score_h)[inx])
                label_res_.extend(np.array(tmp_label_h)[inx])

            time_elapsed = timer() - start

            if save_res:
                scores = np.array(score_res_)
                labels = np.array(label_res_)
                boxes = np.array(box_res_)
                valid_show = scores > cfgs.SHOW_SCORE_THRSHOLD
                scores = scores[valid_show]
                boxes = boxes[valid_show]
                labels = labels[valid_show]
                det_detections_h = draw_for_paper((np.array(img, np.float32) - np.array(cfgs.MXNET_MEAN))/np.array(cfgs.MXNET_STD),
                                                boxes=boxes,
                                                labels=labels,
                                                scores=scores)
                det_detections_h = det_detections_h[:ori_H, :ori_W]
                save_dir = os.path.join(des_folder, cfgs.VERSION)
                tools.mkdir(save_dir)
                cv2.imwrite(save_dir + '/' + img_path.split('/')[-1].split('.')[0] + '_h_s%d_t%f.jpg' %(h_len, cfgs.FAST_RCNN_NMS_IOU_THRESHOLD),
                            det_detections_h)

                view_bar('{} cost {}s'.format(img_path.split('/')[-1].split('.')[0],
                                              time_elapsed), count + 1, len(file_paths))

            else:
                # eval txt
                CLASS_DOTA = NAME_LABEL_MAP.keys()

                # Task2
                write_handle_h = {}
                txt_dir_h = os.path.join('txt_output', cfgs.VERSION + '_h')
                tools.mkdir(txt_dir_h)
                for sub_class in CLASS_DOTA:
                    if sub_class == 'back_ground':
                        continue
                    write_handle_h[sub_class] = open(os.path.join(txt_dir_h, 'Task2_%s.txt' % sub_class), 'a+')

                for i, hbox in enumerate(box_res_):
                    command = '%s %.3f %.1f %.1f %.1f %.1f\n' % (img_path.split('/')[-1].split('.')[0],
                                                                 score_res_[i],
                                                                 hbox[0], hbox[1], hbox[2], hbox[3])
                    write_handle_h[LABEl_NAME_MAP[label_res_[i]]].write(command)

                for sub_class in CLASS_DOTA:
                    if sub_class == 'back_ground':
                        continue
                    write_handle_h[sub_class].close()

            view_bar('{} cost {}s'.format(img_path.split('/')[-1].split('.')[0],
                                          time_elapsed), count + 1, len(file_paths))
            fw.write('{}\n'.format(img_path))
            fw.close()
        os.remove(TMP_FILE)


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = ''

    file_paths = get_file_paths_recursive('/home/omnisky/DataSets/Dota/test/images/images', '.png')
    if cfgs.USE_CONCAT:
        from libs.networks import build_whole_network_Concat
        det_net = build_whole_network_Concat.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                              is_training=False)
    else:
        det_net = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                       is_training=False)
    inference(det_net, file_paths, '/home/omnisky/TF_Codes/horizen_code/tools/demos', 800, 800,
              200, 200, save_res=True)

