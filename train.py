import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import AlexNet
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import tensorflowjs as tfjs

if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class Train:
    def __init__(self):
        self._input_shape = (224, 224, 3)
        self._output_dim = 2

        model = AlexNet(self._input_shape, self._output_dim).get_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=["accuracy"])
        reduce_lro_n_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto',
                                              min_delta=0.0001, cooldown=5, min_lr=1e-10)
        early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

        data_gen = ImageDataGenerator()
        train_it = data_gen.flow_from_directory('data/all/train/', target_size=(224, 224))
        val_it = data_gen.flow_from_directory('data/all/validation/', target_size=(224, 224))

        callbacks_list = [early, reduce_lro_n_plat]

        try:
            model.fit(
                train_it,
                batch_size=32,
                epochs=10000,
                validation_data=val_it,
                callbacks=callbacks_list,
                verbose=1
            )
        except KeyboardInterrupt:
            pass

        model.save_weights("data/model.h5")
        tfjs.converters.save_keras_model(model, "data/model")


if __name__ == '__main__':
    Train()
