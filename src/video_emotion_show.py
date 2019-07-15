#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2019/7/6
@Description:
"""
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from utils.inference import detect_faces
from utils.inference import draw_text, draw_text_lines
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.preprocessor import process_img
import time


"""RAF_DB"""
emotion_model_path = '../trained_models/emotion_models/VGG16_Dense_RAF_20190714.h5'  # 选择模型
emotion_labels = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']
"""FER2013"""
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# emotion_model_path = '../trained_models/emotion_models/fer2013_vgg16_tf.h5'

isgray = False
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=True)
emotion_target_size = emotion_classifier.input_shape[1:3]


def detect_emotion(image):
    bgr_image = image
    if isgray:
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        gray_image = np.squeeze(gray_image)
        gray_image = gray_image.astype('uint8')
        image_origin = gray_image
    else:
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_image = np.squeeze(rgb_image).astype('uint8')
        image_origin = rgb_image
    emotion_offsets = (0, 0)

    faces = detect_faces(face_detection, image_origin)

    if len(faces) > 0:
        for face in faces:
            x1, x2, y1, y2 = apply_offsets(face, emotion_offsets)
            face_image = image_origin[y1:y2, x1:x2]

            face_image = cv2.resize(face_image, emotion_target_size)

            face_image = process_img(face_image)
            face_image = np.expand_dims(face_image, 0)
            if isgray:
                face_image = np.expand_dims(face_image, -1)

            # emotion_prediction = emotion_classifier.predict(gray_face)
            # max_emotion_probability = np.max(emotion_prediction)
            # emotion_label_arg = np.argmax(emotion_prediction)
            # max_emotion_text = emotion_labels[int(emotion_label_arg)]+max_emotion_probability

            emotion_texts = []
            emotion_prediction = emotion_classifier.predict(face_image)[0]
            max_emotion_probability = np.max(emotion_prediction)
            emotion_labels_arg = np.argsort(-emotion_prediction)
            max_emotion_text = emotion_labels[int(emotion_labels_arg[0])]
            for emotion_label_arg in emotion_labels_arg:
                emotion_texts.append(emotion_labels[int(emotion_label_arg)] + ' '
                                     + str('%.2f' % emotion_prediction[int(emotion_label_arg)]))

            if max_emotion_text == 'Angry':
                color = max_emotion_probability * np.asarray((255, 0, 0))
            elif max_emotion_text == 'Sad':
                color = max_emotion_probability * np.asarray((0, 0, 255))
            elif max_emotion_text == 'Happy':
                color = max_emotion_probability * np.asarray((255, 255, 0))
            elif max_emotion_text == 'Surprise':
                color = max_emotion_probability * np.asarray((0, 255, 255))
            else:
                color = max_emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()

            draw_bounding_box(face, bgr_image, color)
            # draw_text(face, rgb_image, max_emotion_text, color, 0, -50, 1, 2)
            draw_text_lines(face, bgr_image, emotion_texts, (255, 255, 255), 0, -50, 0.5, 1)

    return bgr_image


def show_image(window_name, image):
    start = time.time()
    image = detect_emotion(image)
    fps = 1. / (time.time() - start)
    draw_text((10, 30), image, "FPS:" + str(int(fps)), (0, 0, 255))
    cv2.imshow(window_name, image)


def catch_camera(window_name, camera_idx):
    cv2.namedWindow(window_name)

    # 视频来源，可以来自一段已存好的视频，也可以直接来自摄像头
    cap = cv2.VideoCapture(camera_idx)
    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break

        show_image(window_name, frame)

        c = cv2.waitKey(10)
        if c & 0xFF == ord('q') or c & 0xFF == ord('Q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    catch_camera("Emotion Recognize", 0)    # "../images/000201320.avi"
