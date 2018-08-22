from keras import layers, Model
from keras.applications import ResNet50


def build_model(num_classes, input_size):
    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(input_size, input_size, 3), pooling=None)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = layers.Flatten()(x)

    prediction = layers.Dense(num_classes, activation='softmax', name='fc_prediction')(x)
    model = Model(inputs=base_model.input, outputs=prediction)

    return model
