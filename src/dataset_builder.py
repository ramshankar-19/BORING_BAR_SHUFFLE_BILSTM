import numpy as np
from preprocess import preprocess_signals

def build_image_dataset(signal_filepaths, labels):
    images = []
    image_labels = []
    for i, signals in enumerate(signal_filepaths):
        accel_x, accel_y, accel_z, sound_pressure = signals
        img = preprocess_signals(accel_x, accel_y, accel_z, sound_pressure)
        images.append(img)
        image_labels.append(labels[i])
    X = np.array(images)
    y = np.array(image_labels)
    return X, y

# Example batch call:
# files = [(ax, ay, az, s), ...], labels = [0,1,2]
# X, y = build_image_dataset(files, labels)
