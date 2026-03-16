import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)

class DeepFeatureNet(Model):
    def __init__(self, input_shape, num_classes):
        super(DeepFeatureNet, self).__init__()

        self.conv1 = Conv1D(filters=64, kernel_size=50, strides=6, activation='relu', input_shape=input_shape)
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPooling1D(pool_size=8, strides=8)

        self.conv2 = Conv1D(filters=128, kernel_size=8, strides=1, activation='relu')
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPooling1D(pool_size=4, strides=4)

        self.conv3 = Conv1D(filters=128, kernel_size=8, strides=1, activation='relu')
        self.bn3 = BatchNormalization()
        self.pool3 = MaxPooling1D(pool_size=4, strides=4)

        self.flatten = Flatten()
        self.fc1 = Dense(512, activation='relu')
        self.drop1 = Dropout(0.5)
        self.fc2 = Dense(256, activation='relu')
        self.drop2 = Dropout(0.5)
        self.output_layer = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return self.output_layer(x)






