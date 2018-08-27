import cv2
from keras import layers, Model
from keras.applications import ResNet50
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
import numpy as np
import sys
import os

from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    image_size = 224
    model = ResNet50(weights='imagenet')

    args = sys.argv[1:]

    val_dataset = args[0]
    cat_dir = os.path.join(val_dataset, 'cat')
    dog_dir = os.path.join(val_dataset, 'dog')

    for root, dirs, files in os.walk(cat_dir):
        for f in files:
            img_f = os.path.join(cat_dir, f)
            img = image.load_img(img_f, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = model.predict(x)
            print('preds: {}'.format(decode_predictions(preds, top=1)))
