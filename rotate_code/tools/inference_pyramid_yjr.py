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
from libs.box_utils.coordinate_convert import forward_convert, back_forward_convert
from libs.label_name_dict.label_dict import LABEl_NAME_MAP, NAME_LABEL_MAP
from help_utils import tools
# from libs.box_utils import nms
from libs.box_utils import nms_wrapper
from libs.box_utils import coordinate_convert
from libs.configs import cfgs

def get_file_paths_recursive(folder=None, file_ext=None):
    """ Get the absolute path of all files in given folder recursively
    :param folder:
    :param file_ext:
    :return:
    """
    file_list = []
    if folder is None:
        return file_list

    # for dir_path, dir_names, file_names in os.walk(folder):
    #     for file_name in file_names:
    #         if file_ext is None:
    #             file_list.append(os.path.join(dir_path, file_name))
    #             continue
    #         if file_name.endswith(file_ext):
    #             file_list.append(os.path.join(dir_path, file_name))
    file_list = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(file_ext)]

    return file_list


def inference(det_net, file_paths, des_folder, h_len, w_len, h_overlap, w_overlap, save_res=False):

    # if save_res:
    #     assert cfgs.SHOW_SCORE_THRSHOLD >= 0.5, \
    #         'please set score threshold (example: SHOW_SCORE_THRSHOLD = 0.5) in cfgs.py'
    #
    # else:
    #     assert cfgs.SHOW_SCORE_THRSHOLD < 0.005, \
    #         'please set score threshold (example: SHOW_SCORE_THRSHOLD = 0.00) in cfgs.py'

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

    det_boxes_r, det_scores_r, det_category_r = det_net.build_whole_detection_network(input_img_batch=img_batch,
                                                                                      gtboxes_batch=None,
                                                                                      gtboxes_r_batch=None)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = det_net.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        if not os.path.exists('./tmp.txt'):
            fw = open('./tmp.txt', 'w')
            fw.close()

        fr = open('./tmp.txt', 'r')
        pass_img = fr.readlines()
        fr.close()

        for count, img_path in enumerate(file_paths):
            fw = open('./tmp.txt', 'a+')
            if img_path + '\n' in pass_img:
                continue
            start = timer()
            img = cv2.imread(img_path)

            box_res_rotate = []
            label_res_rotate = []
            score_res_rotate = []

            imgH = img.shape[0]
            imgW = img.shape[1]

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

                    a_patch_box_res_rotate = []
                    a_patch_label_res_rotate = []
                    a_patch_score_res_rotate = []
                    for short_size in cfgs.IMG_SHORT_SIDE_LEN:
                        max_len = 1200
                        if h_len < w_len:
                            new_h, new_w = short_size,  min(int(short_size*float(w_len)/h_len), max_len)
                        else:
                            new_h, new_w = min(int(short_size*float(h_len)/w_len), max_len), short_size

                        img_resize = cv2.resize(src_img, (new_h, new_w))

                        det_boxes_r_, det_scores_r_, det_category_r_ = \
                            sess.run(
                                [det_boxes_r, det_scores_r, det_category_r],
                                feed_dict={img_plac: img_resize[:, :, ::-1]}
                            )
                        det_boxes_r_ = forward_convert(det_boxes_r_, False)  # [x,y,w,h,theta]-->[x,y,x,y..,x,y]
                        det_boxes_r_[:, 0::2] *= (w_len/new_w)
                        det_boxes_r_[:, 1::2] *= (h_len/new_h)
                        det_boxes_r_ = back_forward_convert(det_boxes_r_, False)

                        a_patch_box_res_rotate.append(det_boxes_r_)
                        a_patch_score_res_rotate.append(det_scores_r_)
                        a_patch_label_res_rotate.append(det_category_r_)

                    a_patch_box_res_rotate = np.concatenate(a_patch_box_res_rotate, axis=0)
                    a_patch_label_res_rotate = np.concatenate(a_patch_label_res_rotate, axis=0)
                    a_patch_score_res_rotate = np.concatenate(a_patch_score_res_rotate, axis=0)

                    r_threshold = {'roundabout': 0.3, 'tennis-court': 0.3, 'swimming-pool': 0.3, 'storage-tank': 0.2,
                                   'soccer-ball-field': 0.3, 'small-vehicle': 0.3, 'ship': 0.1, 'plane': 0.3,
                                   'large-vehicle': 0.15, 'helicopter': 0.3, 'harbor': 0.1, 'ground-track-field': 0.3,
                                   'bridge': 0.1, 'basketball-court': 0.3, 'baseball-diamond': 0.3}

                    a_patch_box_after_nms = []
                    a_patch_score_after_nms = []
                    a_patch_label_after_nms = []
                    for sub_class in range(1, cfgs.CLASS_NUM + 1):
                        index = np.where(a_patch_label_res_rotate == sub_class)[0]
                        if len(index) == 0:
                            continue
                        tmp_boxes_r = a_patch_box_res_rotate[index]
                        tmp_label_r = a_patch_label_res_rotate[index]
                        tmp_score_r = a_patch_score_res_rotate[index]

                        tmp_boxes_r = np.array(tmp_boxes_r)
                        tmp = np.zeros([tmp_boxes_r.shape[0], tmp_boxes_r.shape[1] + 1])
                        tmp[:, 0:-1] = tmp_boxes_r
                        tmp[:, -1] = np.array(tmp_score_r)

                        try:
                            inx = nms_wrapper.nms_rotate_cpu(boxes=np.array(tmp_boxes_r),
                                                             scores=np.array(tmp_score_r),
                                                             iou_threshold=r_threshold[LABEl_NAME_MAP[sub_class]],
                                                             max_output_size=500)
                        except:
                            # Note: the IoU of two same rectangles is 0, which is calculated by rotate_gpu_nms
                            jitter = np.zeros([tmp_boxes_r.shape[0], tmp_boxes_r.shape[1] + 1])
                            jitter[:, 0] += np.random.rand(tmp_boxes_r.shape[0], ) / 1000
                            inx = nms_wrapper.nms_rotate_gpu(
                                dets=np.array(tmp, np.float32) + np.array(jitter, np.float32),
                                iou_threshold=float(r_threshold[LABEl_NAME_MAP[sub_class]]),
                                max_keep=500,
                                device_id=0)

                        a_patch_box_after_nms.append(np.array(tmp_boxes_r)[inx])
                        a_patch_score_after_nms.append(np.array(tmp_score_r)[inx])
                        a_patch_label_after_nms.append(np.array(tmp_label_r)[inx])
                    a_patch_box_after_nms = np.concatenate(a_patch_box_after_nms, axis=0)
                    a_patch_score_after_nms = np.concatenate(a_patch_score_after_nms, axis=0)
                    a_patch_label_after_nms = np.concatenate(a_patch_label_after_nms, axis=0)

                    if len(a_patch_box_after_nms) > 0:
                        for ii in range(len(a_patch_box_after_nms)):
                            box_rotate = a_patch_box_after_nms[ii]
                            box_rotate[0] = box_rotate[0] + ww_
                            box_rotate[1] = box_rotate[1] + hh_
                            box_res_rotate.append(box_rotate)
                            label_res_rotate.append(a_patch_label_after_nms[ii])
                            score_res_rotate.append(a_patch_score_after_nms[ii])

            box_res_rotate = np.array(box_res_rotate)
            label_res_rotate = np.array(label_res_rotate)
            score_res_rotate = np.array(score_res_rotate)

            box_res_rotate_, label_res_rotate_, score_res_rotate_ = [], [], []

            # r_threshold = {'roundabout': 0.3, 'tennis-court': 0.3, 'swimming-pool': 0.3, 'storage-tank': 0.3,
            #                'soccer-ball-field': 0.3, 'small-vehicle': 0.3, 'ship': 0.3, 'plane': 0.3,
            #                'large-vehicle': 0.3, 'helicopter': 0.3, 'harbor': 0.3, 'ground-track-field': 0.3,
            #                'bridge': 0.3, 'basketball-court': 0.3, 'baseball-diamond': 0.3}
            r_threshold = {'roundabout': 0.1, 'tennis-court': 0.3, 'swimming-pool': 0.1, 'storage-tank': 0.2,
                           'soccer-ball-field': 0.3, 'small-vehicle': 0.2, 'ship': 0.05, 'plane': 0.3,
                           'large-vehicle': 0.1, 'helicopter': 0.2, 'harbor': 0.0001, 'ground-track-field': 0.3,
                           'bridge': 0.0001, 'basketball-court': 0.3, 'baseball-diamond': 0.3}

            for sub_class in range(1, cfgs.CLASS_NUM + 1):
                index = np.where(label_res_rotate == sub_class)[0]
                if len(index) == 0:
                    continue
                tmp_boxes_r = box_res_rotate[index]
                tmp_label_r = label_res_rotate[index]
                tmp_score_r = score_res_rotate[index]

                tmp_boxes_r = np.array(tmp_boxes_r)
                tmp = np.zeros([tmp_boxes_r.shape[0], tmp_boxes_r.shape[1] + 1])
                tmp[:, 0:-1] = tmp_boxes_r
                tmp[:, -1] = np.array(tmp_score_r)

                try:
                    inx = nms_wrapper.nms_rotate_cpu(boxes=np.array(tmp_boxes_r),
                                                     scores=np.array(tmp_score_r),
                                                     iou_threshold=r_threshold[LABEl_NAME_MAP[sub_class]],
                                                     max_output_size=500)
                except:
                    # Note: the IoU of two same rectangles is 0, which is calculated by rotate_gpu_nms
                    jitter = np.zeros([tmp_boxes_r.shape[0], tmp_boxes_r.shape[1] + 1])
                    jitter[:, 0] += np.random.rand(tmp_boxes_r.shape[0], ) / 1000
                    inx = nms_wrapper.nms_rotate_gpu(dets=np.array(tmp, np.float32) + np.array(jitter, np.float32),
                                                     iou_threshold=float(r_threshold[LABEl_NAME_MAP[sub_class]]),
                                                     max_keep=500,
                                                     device_id=0)

                box_res_rotate_.extend(np.array(tmp_boxes_r)[inx])
                score_res_rotate_.extend(np.array(tmp_score_r)[inx])
                label_res_rotate_.extend(np.array(tmp_label_r)[inx])
            time_elapsed = timer() - start

            if save_res:
                scores = np.array(score_res_rotate_)
                labels = np.array(label_res_rotate_)
                boxes = np.array(box_res_rotate_)
                valid_show = scores > cfgs.SHOW_SCORE_THRSHOLD
                scores = scores[valid_show]
                boxes = boxes[valid_show]
                labels = labels[valid_show]
                det_detections_r = draw_box_in_img.draw_rotate_box_cv(
                    np.array(img, np.float32),
                    boxes=boxes,
                    labels=labels,
                    scores=scores)
                save_dir = os.path.join(des_folder, cfgs.VERSION)
                tools.mkdir(save_dir)
                cv2.imwrite(save_dir + '/' + img_path.split('/')[-1].split('.')[0] + '_r_t%f.jpg' %(cfgs.FAST_RCNN_NMS_IOU_THRESHOLD),
                            det_detections_r)
                view_bar('{} cost {}s'.format(img_path.split('/')[-1].split('.')[0],
                                              time_elapsed), count + 1, len(file_paths))
            else:
                # eval txt
                CLASS_DOTA = NAME_LABEL_MAP.keys()
                # Task1
                write_handle_r = {}
                txt_dir_r = os.path.join('txt_output', cfgs.VERSION + '_r')
                tools.mkdir(txt_dir_r)

                for sub_class in CLASS_DOTA:
                    if sub_class == 'back_ground':
                        continue
                    write_handle_r[sub_class] = open(os.path.join(txt_dir_r, 'Task1_%s.txt' % sub_class), 'a+')

                rboxes = coordinate_convert.forward_convert(box_res_rotate_, with_label=False)

                for i, rbox in enumerate(rboxes):
                    command = '%s %.3f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n' % (
                    img_path.split('/')[-1].split('.')[0],
                    score_res_rotate_[i],
                    rbox[0], rbox[1], rbox[2], rbox[3],
                    rbox[4], rbox[5], rbox[6], rbox[7],)
                    write_handle_r[LABEl_NAME_MAP[label_res_rotate_[i]]].write(command)
                for sub_class in CLASS_DOTA:
                    if sub_class == 'back_ground':
                        continue
                    write_handle_r[sub_class].close()

            view_bar('{} cost {}s'.format(img_path.split('/')[-1].split('.')[0],
                                          time_elapsed), count + 1, len(file_paths))
            fw.write('{}\n'.format(img_path))
            fw.close()
        os.remove('./tmp.txt')


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    file_paths = get_file_paths_recursive('/home/omnisky/DataSets/Dota/test/images/images', '.png')
    if cfgs.USE_CONCAT:
        from libs.networks import build_whole_network_Concat
        det_net = build_whole_network_Concat.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                              is_training=False)
    else:
        det_net = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                       is_training=False)
    inference(det_net, file_paths, '/home/omnisky/TF_Codes/semi_rotate/tools/demos', 800, 800,
              200, 200, save_res=False)

