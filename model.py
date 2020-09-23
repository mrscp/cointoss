from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D


class AlexNet:
    def __init__(self, input_shape, output_dim):
        self._input_shape = input_shape
        self._output_dim = output_dim

        self._model = self.build_model()

    def get_model(self):
        return self._model

    def build_model(self):
        model = Sequential()
        # 1st Convolution Layer
        model.add(Conv2D(filters=96, input_shape=self._input_shape, kernel_size=(11, 11), strides=(4, 4), padding="valid"))
        model.add(Activation("relu"))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

        # 2nd Convolution Layer
        model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding="valid"))
        model.add(Activation("relu"))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

        # 3rd Convolution Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="valid"))
        model.add(Activation("relu"))

        # 4th Convolution Layer
        model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="valid"))
        model.add(Activation("relu"))

        # 5th Convolution Layer
        model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="valid"))
        model.add(Activation("relu"))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

        # Passing it to a Fully Connected layer
        model.add(Flatten())
        # 1st Fully Connected Layer
        model.add(Dense(4096, input_shape=(224 * 224 * 3,)))
        model.add(Activation("relu"))
        # Add Dropout to prevent over-fitting
        model.add(Dropout(0.4))

        # 2nd Fully Connected Layer
        model.add(Dense(4096))
        model.add(Activation("relu"))
        # Add Dropout
        model.add(Dropout(0.4))

        # 3rd Fully Connected Layer
        model.add(Dense(1000))
        model.add(Activation("relu"))
        # Add Dropout
        model.add(Dropout(0.4))

        # Output Layer
        model.add(Dense(self._output_dim, activation="softmax"))

        return model



