import os

import cv2
import numpy as np
from keras_applications.imagenet_utils import preprocess_input
from keras_preprocessing import image


def load_clustered_images(images_dir):
    """

    :param images_dir: images dir which is organized as below
                     ├── Ariel_Sharon
                     │   ├── Ariel_Sharon_0006.png
                     │   ├── Ariel_Sharon_0007.png
                     │   ├── Ariel_Sharon_0008.png
                     │   ├── Ariel_Sharon_0009.png
                     │   └── Ariel_Sharon_0010.png
                     |
                     ├── Arnold_Schwarzenegger
                     │   ├── Arnold_Schwarzenegger_0006.png
                     │   ├── Arnold_Schwarzenegger_0007.png
                     │   ├── Arnold_Schwarzenegger_0008.png
                     │   ├── Arnold_Schwarzenegger_0009.png
                     │   └── Arnold_Schwarzenegger_0010.png
                     |
                     ├── Colin_Powell
                     │   ├── Colin_Powell_0006.png
                     │   ├── Colin_Powell_0007.png
    :return:    dict that key is name of each image, value is image files list such as
                {'Ariel_Sharon': ['/data/Ariel_Sharon/Ariel_Sharon_0006.png',
                                  '/data/Ariel_Sharon/Ariel_Sharon_0007.png' ... ] }
    """
    sub_dirs = [x for x in os.walk(images_dir)]

    def _is_image_file(f_name):
        ext = f_name.split(os.extsep)[-1]
        if len(ext) < 1:
            return False
        ext = ext.lower()
        return ext in ['jpg', 'jpeg', 'gif', 'png', 'bmp']

    for i in range(1, len(sub_dirs)):
        base_path, _, f_names = sub_dirs[i]
        label = base_path.split(os.sep)[-1]
        file_paths = [os.path.join(base_path, f) for f in f_names if _is_image_file(f)]
        file_paths.sort()
        yield label, file_paths


def read_img(image_path, target_size, rescale=1):
    img = image.load_img(image_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x *= rescale
    return x


def save_img(im, image_path):
    cv2.imwrite(image_path,im)


def put_txt(im, txt, pos, color):
    im = cv2.putText(im, txt, pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=color, thickness=2)
    return im
