import glob
import os
import string
import time
from os import path
from pathlib import Path

import numpy as np
import scipy.spatial as sp
import tensorflow as tf
from cv2 import cv2
from PIL import Image

import frameExtractor
from handshape_feature_extractor import HandShapeFeatureExtractor

# model
FILE_MODEL = 'multi_person_mobilenet_v1_075_float.tflite'

# Here you can define your traindata_original values
h, w = 200, 200

# adjustments for hand extraction
H_ADJUST = 80
V_ADJUST = 180

# desire comparison score 0<=DESIRE_SCORE<=1
DESIRE_SCORE = 0.5

# resize the image default 0.5
IMAGE_RESIZE = 0.8

# test and train data
test = []
train = []

train_dict = {}

# result alphabet
ALPHABET = list(string.ascii_uppercase)


def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180

    cross = v2[0] * v1[1] - v2[1] * v1[0]
    if cross < 0:
        angle = - angle
    return angle


def crop_video(file_folder):
    # mp4 video 30 fps
    # for each gesture will be displayed 3 second?

    crop_path = os.path.join(os.getcwd(),
                              'traindata' if file_folder == 'traindata_original' else 'test')

    for videoFile in get_video_files(file_folder):
        saving_path = crop_path + "//" + Path(os.path.realpath(videoFile)).stem + ".mp4"
        interpreter = tf.lite.Interpreter(model_path=FILE_MODEL)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        # print('input_details\n', input_details)
        output_details = interpreter.get_output_details()
        # print('output_details\n', output_details)

        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        # Open video file
        cap = cv2.VideoCapture(videoFile)

        # Initialize frame counter
        frame_count = 0
        fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(frames)

        # generate output file
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(saving_path, fourcc, fps, (w, h))

        while True:
            # read one frame of image
            success, img = cap.read()
            if not success:
                break
            # get size of the image frame
            imH, imW, _ = np.shape(img)

            # resize the image default 0.5
            img = cv2.resize(img, (int(imW * IMAGE_RESIZE), int(imH * IMAGE_RESIZE)))

            # get size of the image frame
            imH, imW, _ = np.shape(img)

            # BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # resize to adapt PosNet requirement
            img_resized = cv2.resize(img_rgb, (width, height))
            input_data = np.expand_dims(img_resized, axis=0)

            input_data = (np.float32(input_data) - 128.0) / 128.0

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            hotmaps = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
            offsets = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects

            h_output, w_output, n_KeyPoints = np.shape(hotmaps)
            keypoints = []

            # score for key point
            score = 0

            for i in range(n_KeyPoints):
                hotmap = hotmaps[:, :, i]

                max_index = np.where(hotmap == np.max(hotmap))
                max_val = np.max(hotmap)

                offset_y = offsets[max_index[0], max_index[1], i]
                offset_x = offsets[max_index[0], max_index[1], i + n_KeyPoints]

                pos_y = max_index[0] / (h_output - 1) * height + offset_y
                pos_x = max_index[1] / (w_output - 1) * width + offset_x

                pos_y = pos_y / (height - 1) * imH
                pos_x = pos_x / (width - 1) * imW

                keypoints.append([int(round(pos_x[0])), int(round(pos_y[0]))])

                score = score + 1.0 / (1.0 + np.exp(-max_val))
            score = score / n_KeyPoints

            if score > DESIRE_SCORE:
                # Position of two hand wrists 9, 10
                # key point 10
                cv2.circle(img, (keypoints[9][0], keypoints[9][1]), 0, (0, 255, 0), 0)
                # cv2.circle(img, (keypoints[10][0], keypoints[10][1]), 0, (0, 0, 255), 0)

                # key point connection
                # left arm
                cv2.polylines(img, [np.array([keypoints[5], keypoints[7], keypoints[9]])], False, (0, 255, 0), 0)
                # right arm, will be used for crop image for ASL Finger Spelling
                # cv2.polylines(img, [np.array([keypoints[6], keypoints[8], keypoints[10]])], False, (0, 0, 255), 0)

            # display result
            cv2.imshow('original', img)

            # ===========================================================
            # This section will crop the video and save hand section
            x = (keypoints[10][0] - H_ADJUST) if (keypoints[10][0] - H_ADJUST) > 0 else 0
            y = (keypoints[10][1] - V_ADJUST) if (keypoints[10][1] - V_ADJUST) > 0 else 0
            # CropPing the frame
            crop_frame = img[y:y + h, x:x + w]

            # Saving from the desired frames
            # if 15 <= cnt <= 90:
            #    out.write(crop_frame)

            # save the result video
            out.write(crop_frame)

            cv2.imshow('cropped', crop_frame)
            frame_count += 1
            # ===========================================================

            # press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()


def get_video_files(file_folder):
    if not path.exists(file_folder):
        return []
    return glob.glob(os.path.join(file_folder, '*.mp4'))


def calc_dataset(file_folder):
    frame_count = 0
    model = HandShapeFeatureExtractor.get_instance()
    frame_path = os.path.join(os.getcwd(),
                'traindata_frame' if file_folder == 'traindata' else 'test_frame')

    for videoFile in get_video_files(file_folder):
        frameExtractor.frameExtractor(videoFile, frame_path, frame_count)
        frame_count += 1

    frame_files = glob.glob(os.path.join(frame_path, '*.png'))

    vectors = []
    for frame_file in frame_files:
        img = cv2.cvtColor(cv2.imread(frame_file), cv2.COLOR_BGR2GRAY)
        results = model.extract_feature(img)[0]
        vectors.append(results)
    return vectors


def calc_result(test_vec, train_vec):
    test_data = np.array(test_vec, dtype=float)
    training_data = np.array(train_vec, dtype=float)

    f = open('Results.csv', 'w')

    for each_test in test_data:
        lst = []
        for each_train in training_data:
            lst.append(sp.distance.cosine(each_test, each_train))
            gesture_num = lst.index(min(lst))
        gesture_str = ALPHABET[int(gesture_num/5)]
        f.write(gesture_str + '\n')
    f.close()


def remove_file():
    dir = os.path.join(os.getcwd(), 'test_frame')
    for f in glob.glob(os.path.join(dir, "*")):
        os.remove(f)

    dir = os.path.join(os.getcwd(), 'test')
    for f in glob.glob(os.path.join(dir, "*")):
        os.remove(f)


if __name__ == '__main__':
    for i in range(65, 91):
        train_dict[chr(i)] = []

    remove_file()

    start = time.time()
    # crop_video('traindata_original')
    crop_video('test_original')

    train = calc_dataset('traindata')
    test = calc_dataset('test')

    calc_result(test, train)
    end = time.time()
    print(end - start)

