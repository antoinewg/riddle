import os
from datetime import datetime

from tensorflow.keras import callbacks


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

now = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join(BASE_DIR, "logs/scalars/") + now

tensorboard_callback = callbacks.TensorBoard(
    log_dir=logdir, histogram_freq=1, profile_batch="500,520"
)
