import datetime
import os
import sys
from argparse import ArgumentParser

import keras.backend as K
from keras import layers, Model
from keras.applications import ResNet50
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.optimizers import SGD
from keras_applications.resnet50 import preprocess_input
from keras_preprocessing.image import ImageDataGenerator

LOGS_DIR = os.path.join(os.path.dirname(__file__), '../logs')
NAME_PREFIX = 'dog_vs_cat_resnet50'


# 构建模型
def build_resnet50_model(num_classes, input_size):
    #使用keras提供的, 包含imagenet weight的Reset50做为基础模型, 进行transfer learning
    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(input_size, input_size, 3), pooling='max')

    x = base_model.output
    x = layers.MaxPooling2D(input_shape=base_model.layers[-1].output_shape[1:])(x)

    x = layers.Flatten()(x)

    x = layers.Dropout(0.75)(x)
    x = layers.Dense(1024)(x)
    x = layers.BatchNormalization()(x)
    features = layers.Activation('relu')(x)

    # 使用simoid激活方法, 训练二分分类器
    prediction = layers.Dense(num_classes, activation='sigmoid', name='fc_prediction')(features)
    model = Model(inputs=base_model.input, outputs=prediction)
    return model, base_model


# 回调, 生成模型文件, 使用tensorboard
def create_checkpoint():
    filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = "{}/{}_{}".format(LOGS_DIR, NAME_PREFIX, filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tensor_board = TensorBoard(log_dir=log_dir)
    f_name = '{epoch:02d}_{val_acc:.4f}'
    model_file_path = '{}/{}_{}_{}.h5'.format(log_dir, NAME_PREFIX, f_name, filename)

    checkpoint = ModelCheckpoint(model_file_path, monitor='val_loss', verbose=1, period=1)
    return [checkpoint, tensor_board], model_file_path


def main(train_dataset, val_dataset, epochs, batch_size, lr=0.001, image_size=224, weights_file=None):
    # 训练集数据预处理和图形增广
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rescale=1 / 255.,
        rotation_range=10,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True). \
        flow_from_directory(train_dataset, target_size=(image_size, image_size), class_mode='binary')

    # 验证集数据预处理和图形增广
    val_gen = ImageDataGenerator(
        rescale=1 / 255.,
        rotation_range=10,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True). \
        flow_from_directory(val_dataset, target_size=(image_size, image_size), class_mode='binary')

    model, base_model = build_resnet50_model(train_gen.num_classes, image_size)
    if weights_file is not None:
        print('Loading weights file: {}'.format(weights_file))
        model.load_weights(weights_file, by_name=True, skip_mismatch=True)

    callbacks, model_file_path = create_checkpoint()

    # 选取optimizer, 设置learning rate, 编译模型
    model.compile(optimizer=SGD(lr, momentum=0.9),
                  loss="binary_crossentropy",
                  metrics=['accuracy'])

    # 打印模型结构
    model.summary()

    # 训练模型
    model.fit_generator(train_gen,
                        epochs=epochs,
                        steps_per_epoch=train_gen.samples / batch_size,
                        callbacks=callbacks,
                        validation_data=val_gen,
                        validation_steps=val_gen.samples / batch_size,
                        verbose=1)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('train_dataset', type=str)
    parser.add_argument('val_dataset', type=str)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=float, default=512)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    main(args.train_dataset, args.val_dataset, args.epochs, args.batch_size, args.lr, args.image_size, args.weights)
