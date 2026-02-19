import os
import cv2
import numpy as np

REAL_DIR = "../dataset/real"
FAKE_DIR = "../dataset/fake"
IMG_SIZE = 224

data = []
labels = []

def load_images(folder, label):
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(label)

load_images(REAL_DIR, 0)   # real
load_images(FAKE_DIR, 1)   # fake

X = np.array(data) / 255.0
y = np.array(labels)

print("Total images:", X.shape)
print("Labels:", y.shape)
print("Real:", list(y).count(0))
print("Fake:", list(y).count(1))
