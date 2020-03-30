# -*- coding:utf-8 -*-
import cv2
import numpy as np


def getROI(image_np, boxes_roi, margin_ratio):
    """
    get region of interest
    :param image_np:
    :param boxes_roi:
    :param margin_ratio:
    :return: region of interest
    """
    (left, right, top, bottom) = boxes_roi[0]
    width = right - left
    height = bottom - top
    margin_x = margin_ratio * width
    margin_y = margin_ratio * height
    img_roi = image_np[
        int(top - margin_y): int(bottom + margin_y),
        int(left - margin_x): int(right + margin_x)]
    return img_roi


def drawBoxOfROI(scores_roi, boxes_roi, padding_ratio,
                 margin_ratio, im_width, im_height, image_np):
    """
    check if there is hand, and generate ROI and padded image
    :param scores_roi:
    :param boxes_roi:
    :param padding_ratio:
    :param margin_ratio:
    :param im_width:
    :param im_height:
    :param image_np:
    :return: flag, ROI, padded image
    """
    try:
        (left, right, top, bottom) = boxes_roi[0]
        padding_x = padding_ratio * (right - left)
        padding_y = padding_ratio * (bottom - top)
        margin_x = margin_ratio * (right - left)
        margin_y = margin_ratio * (bottom - top)
        padding_left = max(int(left - padding_x), 0)
        padding_top = max(int(top - padding_y), 0)
        padding_right = min(int(right + padding_x), im_width - 1)
        padding_bottom = min(int(bottom + padding_y), im_height - 1)
        margin_left = padding_left
        margin_top = max(int(top - margin_y), 0)
        margin_right = padding_right
        margin_bottom = padding_bottom
        b_have_hand = True
        image_clone_1 = image_np.copy()
        image_clone_2 = image_np.copy()
        image_extend = image_clone_1[margin_top:margin_bottom,
                                     margin_left:margin_right]
        image_roi = image_clone_2[padding_top:padding_bottom,
                                  padding_left:padding_right]
        cv2.rectangle(image_np, (margin_left, margin_top),
                      (margin_right, margin_bottom), (0, 255, 255), 2, 1)
        cv2.putText(image_np, str(float('%.2f' % scores_roi[0])),
                    (int(margin_left), int(margin_top)-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 100), 2)
        print('here')
        return b_have_hand, image_roi, image_extend
    except Exception as e:
        b_have_hand = False
        image_roi = np.zeros((150, 150, 3), np.uint8)
        return b_have_hand, image_roi, image_roi

