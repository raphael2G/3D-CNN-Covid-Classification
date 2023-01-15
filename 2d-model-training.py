import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras import layers


(train_dataset, test_dataset), info = tfds.load('mnist', split=['train', 'test'], with_info=True, as_supervised=True, shuffle_files=True)
tfds.as_dataframe(train_dataset.take(5), info)

model = tf.Sequential([
    tf.keras.layers.Input([28, 28, 3]), 
    tf.keras.layers.Conv2D(3, 3, (3,3)), 
])

model = tf.keras.Functional(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.GlobalAveragePositioning()
        layers.Dense(num_classes, activation="softmax"),
    ]
)

