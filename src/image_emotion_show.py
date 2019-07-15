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

from skimage import io

from utils.inference import load_image, detect_faces, apply_offsets
from utils.preprocessor import process_img

image_path = "../images/sad3.jpg"     # 选择测试图片

"""RAF_DB"""
emotion_model_path = '../trained_models/emotion_models/VGG16_Dense_RAF_20190714.h5'      # 选择加载模型
emotion_labels = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']
"""FER2013"""
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# emotion_model_path = '../trained_models/emotion_models/fer2013_vgg16_tf.h5'

isgray = False
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=True)
emotion_target_size = emotion_classifier.input_shape[1:3]

rgb_image = load_image(image_path, color_mode='rgb')
gray_image = load_image(image_path, color_mode='grayscale')
if isgray:
    # gray_image = rgb2gray(rgb_image)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')
    face_image = gray_image
else:
    rgb_image = np.squeeze(rgb_image).astype('uint8')
    face_image = rgb_image

emotion_offsets = (0, 0)

# face = gray_image

faces = detect_faces(face_detection, face_image)
face_crop = face_image
if len(faces) != 0:
    face = faces[0]
    x1, x2, y1, y2 = apply_offsets(face, emotion_offsets)
    face_crop = face_image[y1:y2, x1:x2]

face_image = cv2.resize(face_crop, emotion_target_size)
face_image = process_img(face_image)
face_image = np.expand_dims(face_image, 0)
if isgray:
    face_image = np.expand_dims(face_image, -1)

emotion_values = emotion_classifier.predict(face_image)
emotion_label_arg = np.argmax(emotion_values)
emotion_text = emotion_labels[int(emotion_label_arg)]

# draw_bounding_box(face, rgb_image, color)
# draw_text(face, rgb_image, emotion_text, color, 0, -50, 1, 2)

print("The Expression is %s" % emotion_text)
plt.rcParams['figure.figsize'] = (13.5, 5.5)
axes = plt.subplot(1, 3, 1)
plt.imshow(face_crop/255.)
plt.xlabel('Input Image', fontsize=16)
axes.set_xticks([])
axes.set_yticks([])
plt.tight_layout()
plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)
plt.subplot(1, 3, 2)
ind = 0.1 + 0.6 * np.arange(len(emotion_labels))  # the x locations for the groups
width = 0.4  # the width of the bars: can also be len(x) sequence
color_list = ['red', 'orangered', 'darkorange', 'limegreen', 'darkgreen', 'royalblue', 'navy']
for i in range(len(emotion_labels)):
    plt.bar(ind[i], emotion_values[0][i], width, color=color_list[i])
plt.title("Classification results ", fontsize=20)
plt.xlabel(" Expression Category ", fontsize=16)
plt.ylabel(" Classification Score ", fontsize=16)
plt.xticks(ind, emotion_labels, rotation=45, fontsize=14)
axes = plt.subplot(1, 3, 3)
emojis_img = io.imread('../images/emojis/%s.png' % emotion_text)
plt.imshow(emojis_img)
plt.xlabel('Emoji Expression', fontsize=16)
axes.set_xticks([])
axes.set_yticks([])
plt.tight_layout()
# show emojis
plt.show()
# plt.savefig(os.path.join('images/results/l.png'))
# plt.close()
