import os
import sys
from argparse import ArgumentParser

import cv2
from keras.engine.saving import load_model

from train.utils import read_img, save_img, put_txt

image_ext = ['jpg', 'jpeg', 'png', 'gif']

input_size = 224

# 加载训练好的模型
def load(weight_path):
    model = load_model(weight_path)
    model.summary()
    return model


def main(weight_path, dataset, out_dir, generate_output_image=True):
    model = load(weight_path=weight_path)

    count = 0
    results = []
    for root, dirs, files in os.walk(dataset):
        for im_f in files:
            split = im_f.split(os.extsep)
            fname = split[0]
            ext = split[-1]
            count += 1
            if count % 500 == 0:
                print("{} done.".format(count))
            if ext in image_ext:
                image_file = os.path.join(dataset, im_f)
                im = read_img(image_file, (input_size, input_size), rescale=1 / 255.) #图片预处理

                pred = model.predict(im)[0]
                f_name = os.path.join(out_dir, '{}'.format(im_f))
                img = cv2.imread(image_file)
                if generate_output_image:
                    txt1 = 'cat: {0:.06f}%'.format((1 - pred[0]) * 100)
                    txt2 = 'dog: {0:.06f}%'.format(pred[0] * 100)
                    img = put_txt(img, txt1, (10, 30), (0, 255, 0))
                    img = put_txt(img, txt2, (10, 60), (0, 255, 0))
                    save_img(img, f_name)

                probability = pred[0]
                # print('{}: {}'.format(im_f, probability))
                results.append((int(fname), probability))

    results.sort(key=lambda x: x[0]) #按照id进行排序

    #生成submisison.csv
    with open(os.path.join(out_dir, 'submission.csv'), 'w', encoding='utf-8') as f_submission:
        f_submission.write('id,label\n')
        for id, label in results:
            f_submission.write('{},{}\n'.format(str(id), str(label)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('weight_path', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('--out_dir', type=str, default='.')
    parser.add_argument('--generate_output_image', type=bool, default=True)
    args = parser.parse_args(sys.argv[1:])
    main(args.weight_path, args.dataset, args.out_dir, args.generate_output_image)
