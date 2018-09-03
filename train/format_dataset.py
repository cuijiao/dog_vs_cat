import os
import random
import sys
from argparse import ArgumentParser
from shutil import copyfile

IMAGE_FILES_EXT = ['jpg', 'jpeg', 'png', 'gif']


def split_train_val(target_dir, val_ratio=0.05):
    parent_dir = os.path.dirname(target_dir)
    print(parent_dir)
    print(target_dir)
    curr_dir = target_dir.split(os.sep)[-1]
    curr_dir = os.path.join(parent_dir, '{}_train_val_split'.format(curr_dir))
    if not os.path.exists(curr_dir):
        os.mkdir(curr_dir)
    train_dir = os.path.join(curr_dir, 'train')
    val_dir = os.path.join(curr_dir, 'val')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)

    for root, dirs, _ in os.walk(target_dir):
        for dir_ in dirs:
            print(dir_)
            copy_files(target_dir, dir_, train_dir, val_dir, val_ratio)


def copy_files(target_dir, dir_, train_dir, val_dir, val_ratio):
    files_dir = os.path.join(target_dir, dir_)

    tar_train_dir = os.path.join(train_dir, dir_)
    tar_val_dir = os.path.join(val_dir, dir_)
    if not os.path.exists(tar_train_dir):
        os.mkdir(tar_train_dir)
    if not os.path.exists(tar_val_dir):
        os.mkdir(tar_val_dir)

    for _, _, files in os.walk(files_dir):
        print('Split dataset: {}'.format(dir_))
        files_len = len(files)
        val_len = int(val_ratio * files_len)

        print('train_len : {}'.format(files_len - val_len))
        print('val_len : {}'.format(val_len))

        random.shuffle(files)
        val_files = files[0:val_len]
        train_files = files[val_len:]

        for t in train_files:
            f_name = os.path.join(files_dir, t)
            t_name = os.path.join(tar_train_dir, t)
            copyfile(f_name, t_name)

        for v in val_files:
            f_name = os.path.join(files_dir, v)
            t_name = os.path.join(tar_val_dir, v)
            copyfile(f_name, t_name)


def main(dog_vs_cat_images_dir, target_dir, val_ratio):
    if target_dir.endswith(os.sep):
        target_dir = target_dir[:len(target_dir) - 1]
    dog_dir = os.path.join(target_dir, 'dog')
    cat_dir = os.path.join(target_dir, 'cat')
    if not os.path.exists(dog_dir):
        os.mkdir(dog_dir)
    if not os.path.exists(cat_dir):
        os.mkdir(cat_dir)

    dog_count = 0
    cat_count = 0

    for root, dirs, files in os.walk(dog_vs_cat_images_dir):
        for file in files:
            print('Processing: {}'.format(file))
            ext = file.split(os.extsep)[-1]
            if ext.lower() in IMAGE_FILES_EXT:
                file_path = os.path.join(dog_vs_cat_images_dir, file)
                if file.startswith('dog'):
                    f_name = '{0:08d}'.format(dog_count)
                    tar_path = os.path.join(dog_dir, '{}.{}'.format(f_name, ext))
                    dog_count += 1
                elif file.startswith('cat'):
                    f_name = '{0:08d}'.format(cat_count)
                    tar_path = os.path.join(cat_dir, '{}.{}'.format(f_name, ext))
                    cat_count += 1
                else:
                    print('Do not know what this image is of {}'.format(file))
                    continue

                copyfile(file_path, tar_path)

    print('Dog images: {}'.format(dog_count))
    print('Cat images: {}'.format(cat_count))

    split_train_val(target_dir, val_ratio=val_ratio)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dog_vs_cat_images_dir', type=str)
    parser.add_argument('target_dir', type=str)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    args = parser.parse_args(sys.argv[1:])
    main(args.dog_vs_cat_images_dir, args.target_dir, args.val_ratio)