from __future__ import print_function
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import ImageSerializer
from .models import Image

from .common.mva19 import Estimator, preprocess
import numpy as np
import cv2
import time
import argparse
import tensorflow as tf
from .utils import detector_utils as detector_utils
from .utils import recognizer_utils as recognizer_utils
import os
import numpy as np
from keras.models import *
import time
import shutil


# Links for each sign language picture to online image server
numbers = {
    '1': 'https://i.ibb.co/R40rKGC/1.jpg',
    '2': 'https://i.ibb.co/HNLRcDw/2.jpg',
    '3': 'https://i.ibb.co/W27Gr39/3.jpg',
    '4': 'https://i.ibb.co/r46sRWV/4.jpg',
    '5': 'https://i.ibb.co/zFknqPd/5.jpg',
    '6': 'https://i.ibb.co/CVpVSPq/6.jpg',
    '7': 'https://i.ibb.co/wC33gFc/7.jpg',
    '8': 'https://i.ibb.co/jv3Lkmp/8.jpg',
    '9': 'https://i.ibb.co/tcGzdYc/9.jpg',
}

alphabets = {
    'a': 'https://i.ibb.co/TLkkghg/A.png',
    'b': 'https://i.ibb.co/BNsLzvS/B.png',
    'c': 'https://i.ibb.co/fGKzr5F/C.png',
    'd': 'https://i.ibb.co/G962j3F/D.png',
    'e': 'https://i.ibb.co/JKbBPSX/E.png',
    'f': 'https://i.ibb.co/ZBCqD8V/F.png',
    'g': 'https://i.ibb.co/9Hjc9yx/G.png',
    'h': 'https://i.ibb.co/CmPHBzP/H.png',
    'i': 'https://i.ibb.co/hK11HvT/I.png',
    'j': 'https://i.ibb.co/nLHs7dt/J.png',
    'k': 'https://i.ibb.co/x2RGWxR/K.png',
    'l': 'https://i.ibb.co/QPR3fD4/L.png',
    'm': 'https://i.ibb.co/Z14MyPC/M.png',
    'n': 'https://i.ibb.co/HYktXw2/N.png',
    'o': 'https://i.ibb.co/2d49M00/O.png',
    'p': 'https://i.ibb.co/dPr2Hf6/P.png',
    'q': 'https://i.ibb.co/RSK3GP9/Q.png',
    'r': 'https://i.ibb.co/VDZpskk/R.png',
    's': 'https://i.ibb.co/C55gkCQ/S.png',
    't': 'https://i.ibb.co/b1pM20F/T.png',
    'u': 'https://i.ibb.co/h2c83Dd/U.png',
    'v': 'https://i.ibb.co/0hMMH4h/V.png',
    'w': 'https://i.ibb.co/2PPf224/W.png',
    'x': 'https://i.ibb.co/dcQ32kj/X.png',
    'y': 'https://i.ibb.co/DC2G3HY/Y.png',
    'z': 'https://i.ibb.co/rxhfVdt/Z.png',
}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
map_characters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
# map_characters = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'L'}

generating_dataset = False

refersh_data = True

letter = 'I'
idx = 0

def get_bbox(img_src):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=1,
        help='Max number of hands to detect.')
    args = parser.parse_args()
    if img_src is None:
        print('Load Fail')
    else:
        boxes, scores = detector_utils.detect_objects(
            img_src, detection_graph, sess)
        boxes_to_recog, scores_to_show = detector_utils.draw_box_on_image(
            args.num_hands, score_thresh, scores, boxes,
            img_src.shape[1], img_src.shape[0], img_src)
        boxes_roi = boxes_to_recog
        if len(boxes_roi) > 0:
            return True
        else:
            return False

def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(encoded_data.decode('base64'), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@api_view(['GET', 'POST'])
@renderer_classes([JSONRenderer, ])
@parser_classes([MultiPartParser, FormParser])
def get_classifier(request):
    model_file = "./models/mobnet4f_cmu_adadelta_t1_model.pb"
    input_layer = "input_1"
    output_layer = "k2tfout_0"

    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    score_thresh = 0.1

    stride = 4
    boxsize = 224

    estimator = Estimator(model_file, input_layer, output_layer)

    # json_file = open('rec_model.json', 'r')
    json_file = open('rec_model.json', 'r')
    rec_model_json = json_file.read()
    json_file.close()
    rec_model = model_from_json(rec_model_json)
    # rec_model.load_weights("rec_model_4epochs.h5")
    rec_model.load_weights("rec_model_17epochs.h5")
    print("Loaded rec model from disk")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.4)

    paused = True
    delay = {False: 1, True: 0}

    k = 0

    if request.method == 'POST':
        print(request.POST)
        img_uri = str(request.POST.getlist("uri")[0])
        frame = data_uri_to_cv2_img(img_uri)
        #frame = cv2.imread('img path')
        tic = time.time()
        if get_bbox(frame):

            crop_res = cv2.resize(frame, (boxsize, boxsize))
            img, pad = preprocess(crop_res, boxsize, stride)

            tic = time.time()
            hm = estimator.predict(img)
            dt = time.time() - tic

            hm = cv2.resize(hm, (0, 0), fx=stride, fy=stride)
            bg = cv2.normalize(hm[:, :, -1], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            viz = cv2.normalize(np.sum(hm[:, :, :-1], axis=2), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

            if generating_dataset:
                cv2.imwrite('cust_data/{}/{}.jpg'.format(letter, idx), bg)
                print(idx)

            cv2.imwrite('msk/test/{}.jpg'.format(idx), bg)
            cv2.imwrite('ori/test/{}.jpg'.format(idx), crop_res)

            im = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
            im = cv2.resize(im, (100, 100))
            im = np.expand_dims(im, axis=0)
            result = rec_model.predict(im)
            # Gives the result
            result_letter = map_characters[np.argmax(result[0])]
            frame = cv2.putText(frame, str(result_letter) + ' FPS:{}'.format(int(1/dt)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

            idx += 1
            return Response({'content': str(result_letter)})
        else:
            bg = np.zeros((224, 224), np.int8)
            viz = np.zeros((224, 224), np.int8)
            viz.fill(255)
            dt = time.time() - tic
            frame = cv2.putText(frame, 'MOVE HAND TO CENTER PLEASE FPS:{}'.format(int(1/dt)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
            return Response({'content': 'Please move hand to the center'})

    return Response()


@api_view(['GET'])
@renderer_classes([JSONRenderer, ])
def get_tutorial(request):
    """ API Endpoint for tutorial """
    return Response({
        'numbers': numbers,
        'alphabets': alphabets
    })

@api_view(['POST'])
@renderer_classes([JSONRenderer, ])
def search(request):
    print(request.POST)
    searchkey = str(request.POST.getlist("key")[0])
    searchkey = searchkey.lower()
    res = []
    for c in searchkey:
        if c in numbers:
            temp = {c: numbers[c]}
            res.append(temp)
        else:
            temp = {c: alphabets[c]}
            res.append(temp)

    return Response(res)