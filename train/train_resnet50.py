import datetime
import os
import sys
from argparse import ArgumentParser

import keras.backend as K
from keras import layers, Model
from keras.applications import ResNet50
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.optimizers import Adam, SGD
from keras_applications.resnet50 import preprocess_input
from keras_preprocessing.image import ImageDataGenerator


LOGS_DIR = os.path.join(os.path.dirname(__file__), '../logs')
NAME_PREFIX = 'dog_vs_cat_resnet50'


class OnEpochEnd(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))

        print('Optimizer: {}, Epoch is {}, origin lr: {}, current lr is {}'.format(
            self.model.optimizer.__class__.__name__, epoch, K.eval(lr), K.eval(lr_with_decay)))


def build_resnet50_model(num_classes, input_size):
    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(input_size, input_size, 3), pooling=None)

    x = base_model.output
    x = layers.MaxPooling2D(input_shape=base_model.layers[-1].output_shape[1:])(x)

    x = layers.Flatten()(x)

    x = layers.Dense(1024)(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1])(x)
    x = layers.Dropout(0.75)(x)

    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    features = layers.PReLU(shared_axes=[1])(x)

    prediction = layers.Dense(num_classes, activation='softmax', name='fc_prediction')(features)
    model = Model(inputs=base_model.input, outputs=prediction)
    return model, base_model


def create_checkpoint():
    filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = "{}/{}_{}".format(LOGS_DIR, NAME_PREFIX, filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tensor_board = TensorBoard(log_dir=log_dir)
    f_name = '{epoch:02d}_{val_acc:.4f}'
    model_file_path = '{}/{}_{}_{}.h5'.format(log_dir, NAME_PREFIX, f_name, filename)

    checkpoint = ModelCheckpoint(model_file_path, monitor='val_loss', verbose=1, period=1)
    return [checkpoint, tensor_board, OnEpochEnd()], model_file_path


def main(train_dataset, val_dataset, epochs, batch_size, lr=0.001, image_size=224, weights_file=None):
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True). \
        flow_from_directory(train_dataset, target_size=(image_size, image_size))

    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True). \
        flow_from_directory(val_dataset, target_size=(image_size, image_size))

    model, base_model = build_resnet50_model(train_gen.num_classes, image_size)
    if weights_file is not None:
        print('Loading weights file: {}'.format(weights_file))
        model.load_weights(weights_file, by_name=True, skip_mismatch=True)

    callbacks, model_file_path = create_checkpoint()

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(lr),
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    model.summary()
    model.fit_generator(train_gen,
                        epochs=50,
                        steps_per_epoch=train_gen.samples / batch_size,
                        callbacks=callbacks,
                        validation_data=val_gen,
                        validation_steps=val_gen.samples / batch_size,
                        verbose=1)

    for layer in base_model.layers:
        layer.trainable = True

    model.compile(optimizer=SGD(lr / 10., momentum=0.9, decay=0.001),
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    model.summary()
    model.fit_generator(train_gen,
                        initial_epoch=51,
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