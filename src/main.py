from datetime import datetime

import tensorflow as tf
from tensorflow.keras import callbacks

import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

now = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join(BASE_DIR, "logs/scalars/") + now
tensorboard_callback = callbacks.TensorBoard(
    log_dir=logdir, histogram_freq=1, profile_batch="500,520"
)


# 1/ Load dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2/ Build model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ]
)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

# 3/ Train model
model.fit(x_train, y_train, epochs=20, callbacks=[tensorboard_callback])

# 4/ Evaluate model
model.evaluate(x_test, y_test, verbose=2)
