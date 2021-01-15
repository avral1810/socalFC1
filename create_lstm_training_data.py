import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from get_keys import key_check, keys_to_output_movement, keys_to_output_action

from grab_screen import grab_screen

sys.path.append("..")

from utils import label_map_util

MODEL_NAME = 'fifa_graph2'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')
NUM_CLASSES = 3

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


# ## Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


file_name = 'rnn/training_data' + str(int(time.time())) + '.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        feature_vector = detection_graph.get_tensor_by_name(
            "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/Relu6:0")

        for i in list(range(3))[::-1]:
            print(i + 1)
            time.sleep(1)

        paused = True
        while True:
            if not paused:
                keys = key_check()
                print('keys: ' + str(keys))
                if not keys:
                    continue
                screen = grab_screen(region=None)
                image_np = cv2.resize(screen, (900, 400))
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (rep) = sess.run(
                    [feature_vector],
                    feed_dict={image_tensor: image_np_expanded})

                # save output
                output_movement = keys_to_output_movement(keys)
                output_action = keys_to_output_action(keys)
                training_data.append([rep, output_movement, output_action])
                if len(training_data) % 100 == 0:
                    print(len(training_data))
                    np.save(file_name, training_data)

            keys = key_check()
            if 'P' in keys:
                if paused:
                    paused = False
                    print('unpaused!')
                    time.sleep(1)
                else:
                    print('Pausing!')
                    paused = True
                    time.sleep(1)
            elif 'O' in keys:
                print('Quitting!')
                break
