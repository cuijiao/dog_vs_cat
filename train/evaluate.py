import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
from keras.engine.saving import load_model

sys.path.append('/Users/jcui/MachineLearning/dog_vs_cat')
from train.utils import read_img, save_img, put_txt

image_ext = ['jpg', 'jpeg', 'png', 'gif']

classes = ['cat', 'dog']


def load(weight_path):
    model = load_model(weight_path)
    model.summary()
    return model


def main(weight_path, dataset, out_dir):
    model = load(weight_path=weight_path)
    count = 1
    for root, dirs, files in os.walk(dataset):
        for im_f in files:
            ext = im_f.split(os.extsep)[-1]
            if ext in image_ext:
                print('Processing: {}'.format(im_f))
                image_file = os.path.join(dataset, im_f)
                im = read_img(image_file, (224, 224), rescale=1 / 255.)
                pred = model.predict(im)
                print(pred)
                idx = np.argmax(pred, axis=1)
                label = classes[idx[0]]
                fname = '{0:6d}'.format(count)
                count += 1
                f_name = os.path.join(out_dir, '{}_{}.{}'.format(fname, label, ext))
                img = cv2.imread(image_file)
                txt1 = 'cat: {0:.06f}%'.format(pred[0][0] * 100)
                txt2 = 'dog: {0:.06f}%'.format(pred[0][1] * 100)
                img = put_txt(img, txt1, (10, 30), (0, 255, 0))
                img = put_txt(img, txt2, (10, 60), (0, 255, 0))
                save_img(img, f_name)
            else:
                continue


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('weight_path', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('--out_dir', type=str)
    args = parser.parse_args(sys.argv[1:])
    main(args.weight_path, args.dataset, args.out_dir)
