
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

from direct_keys import *

from get_keys import key_check
from grab_screen import grab_screen
from utils import label_map_util

from tensorflow.keras.models import load_model

sys.path.append("..")
MODEL_NAME = 'fifa_graph2'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 3

def take_action(movement_index, action_index):
    movement = [[uparrow], [downarrow], [leftarrow], [rightarrow], []]
    action = [[spacebar], [W], [Q], [F], []]
    for index in movement[movement_index]:
        print("move",index)
        PressKey(index)
    for index in action[action_index]:
        PressKey(index)
        print("action", index)
    time.sleep(0.2)
    for index in movement[movement_index]:
        ReleaseKey(index)
    for index in action[action_index]:
        ReleaseKey(index)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

steps_of_history = 10
input_window = np.zeros(shape=(steps_of_history, 128))

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        feature_vector = detection_graph.get_tensor_by_name(
            "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/Relu6:0")
        for i in range(0, steps_of_history):
            screen = grab_screen(region=None)
            screen = screen[20:1000, :1910]
            image_np = cv2.resize(screen, (900, 400))
            image_np_expanded = np.expand_dims(image_np, axis=0)

            rep = sess.run([feature_vector], feed_dict={image_tensor: image_np_expanded})
            input_window[i, :] = np.array(rep).reshape(-1, 128)

print('starting to play...')

play = 1
last_time = time.time()
frames_count = 0

with tf.compat.v1.Session(graph=detection_graph).as_default() as sess:
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    feature_vector = detection_graph.get_tensor_by_name(
        "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/Relu6:0")
    paused = True
    g1 = sess
    g2 = tf.Graph()
    model_movement = load_model('./fifa_models2/model_movement')
    model_action = load_model('./fifa_models2/model_action')
    while True:
        if not paused:
            screen = grab_screen(region=None)
            image_np = cv2.resize(screen, (900, 400))
            image_np_expanded = np.expand_dims(image_np, axis=0)

            with detection_graph.as_default():
                (rep) = sess.run([feature_vector], feed_dict={image_tensor: image_np_expanded})
                input_window[:-1, :] = input_window[1:, :]
                input_window[-1, :] = np.array(rep).reshape(-1, 128)
            Y_movement = model_movement.predict(input_window.reshape(-1, 10, 128))
            movement_index = np.argmax(Y_movement)
            Y_action = model_action.predict(input_window.reshape(-1, 10, 128))
            action_index = np.argmax(Y_action)
            print(action_index)

            if play == 1:
                take_action(movement_index, action_index)

            current_time = time.time()
            if current_time - last_time >= 1:
                print('{} frames per second'.format(frames_count))
                last_time = current_time
                frames_count = 0
            else:
                frames_count = frames_count + 1

        keys = key_check()
        if 'P' in keys:
            if paused:

                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                cv2.destroyAllWindows()
                time.sleep(1)
        elif 'O' in keys:
            print('Quitting!')
            cv2.destroyAllWindows()
            break
