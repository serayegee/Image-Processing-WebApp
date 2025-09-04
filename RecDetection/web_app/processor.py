import cv2
import numpy as np

# Görüntüyü griye çevirme
def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Sabit eşikleme ile ikili görüntü üretme
def binary_threshold(image, thresh=127):
    gray = to_gray(image)
    _, binary = cv2.binary_threshold(gray,thresh, 255, cv2.THRESH_BINARY)
    return binary

# Adaptif eşikleme
def adaptive_threshold(image, block_size=11, C=2):
    gray = to_gray(image)
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )
    return adaptive

# Gürültü temizleme
def clean_noise(image, kernel_size=3):
    kernel=np.ones((kernel_size, kernel_size), np.uint8)
    cleaned=cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return cleaned

# Kontur bulma
def find_contours(binary_image):
    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours

# Kontur etrafına dikdörtgen
def draw_bounding_boxes(image, contours, color=(0,255,0), thickness=2):
    annotated=image.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(annotated, (x,y), (x+w,y+h), color, thickness)
    return annotated
