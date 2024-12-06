import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_images(data_dir, img_size=(128, 128)):
    images = []
    labels = []
    label_map = {'sedimentary': 0, 'igneous': 1, 'metamorphic': 2}  # Example label map
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                if image_name.endswith('.jpg') or image_name.endswith('.png'):
                    img_path = os.path.join(folder_path, image_name)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, img_size)
                    images.append(img)
                    labels.append(label_map[folder])
    
    images = np.array(images) / 255.0  # Normalize images
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=len(label_map))  # One-hot encode labels
    return images, labels

def split_data(images, labels, test_size=0.2):
    return train_test_split(images, labels, test_size=test_size, random_state=42)
