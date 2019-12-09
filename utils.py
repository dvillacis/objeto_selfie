import cv2
import numpy as np
from PIL import Image, ImageEnhance

# Convert raw matrix into an image and change color system to RGB
def decode_image_from_raw_bytes(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_image(img_path):
    img = Image.open(img_path)
    img.load()
    img = np.asarray(img, dtype=np.uint8)
    return img

def save_image(img, img_path, img_name):
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(0.9)
    img.save(img_path+'/'+img_name+'.png')

