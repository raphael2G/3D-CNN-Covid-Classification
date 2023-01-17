import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras import layers


# create the data
(train_dataset, test_dataset), info = tfds.load('mnist', split=['train', 'test'], with_info=True, as_supervised=True, shuffle_files=True)
tfds.as_dataframe(train_dataset.take(5), info)


# create the model
model = tf.keras.Sequential([
    tf.keras.layers.Input([28, 28, 3]), 
    tf.keras.layers.Rescaling(scale=1./255),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), 
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'), 
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=tf.keras.metrics.SparseCategoricalAccuracy()
             )
model.summary()

model.fit(train_dataset.batch(16), 1)







