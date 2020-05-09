import tensorflow as tf

from build_model import build_model
from load_data import load_data
from setup_tensorboard import tensorboard_callback

EPOCHS = 10

# 1/ Load dataset
x_train, x_test, y_train, y_test = load_data()

# 2/ Build model
model = build_model()

# 3/ Train model
model.fit(x_train, y_train, epochs=EPOCHS, callbacks=[tensorboard_callback])

# 4/ Evaluate model
model.evaluate(x_test, y_test, verbose=2)
