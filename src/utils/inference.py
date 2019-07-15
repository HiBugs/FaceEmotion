import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

def load_image(image_path, grayscale=False, color_mode='rgb', target_size=None):
    pil_image = image.load_img(image_path, grayscale, color_mode, target_size)
    return image.img_to_array(pil_image)

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=1, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

def draw_text_lines(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=1, thickness=2):
    x, y = coordinates[:2]
    x_start, y_start = x + x_offset, y + y_offset
    dy = 20
    for i, line in enumerate(text):
        if i == 0:
            new_font_scale = font_scale + 0.5
            new_thickness = thickness + 1
        else:
            new_font_scale = font_scale
            new_thickness = thickness
        y = y_start + i * dy
        cv2.putText(image_array, line, (x_start, y), cv2.FONT_HERSHEY_SIMPLEX,
                    new_font_scale, color, new_thickness, cv2.LINE_AA)

def get_colors(num_classes):
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    colors = np.asarray(colors) * 255
    return colors

