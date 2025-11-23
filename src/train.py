import numpy as np
from dataset_builder import build_image_dataset
from shuffle_bilstm import build_shuffle_bilstm
import tensorflow as tf


# --- Simulate dataset: 10 samples/class (for testing) ---
signal_len = 4096
signals_train = []
labels_train = []

for i in range(30):
    if i < 10:            # Stable
        label = 0
        ax = np.sin(2*np.pi*80*np.linspace(0,1,signal_len)) + 0.1*np.random.randn(signal_len)
        ay = np.sin(2*np.pi*120*np.linspace(0,1,signal_len)) + 0.1*np.random.randn(signal_len)
        az = np.sin(2*np.pi*200*np.linspace(0,1,signal_len)) + 0.1*np.random.randn(signal_len)
        sp = np.sin(2*np.pi*340*np.linspace(0,1,signal_len)) + 0.1*np.random.randn(signal_len)
    elif i < 20:          # Transition
        label = 1
        ax = np.sin(2*np.pi*160*np.linspace(0,1,signal_len)) + 0.2*np.random.randn(signal_len)
        ay = np.sin(2*np.pi*180*np.linspace(0,1,signal_len)) + 0.2*np.random.randn(signal_len)
        az = np.sin(2*np.pi*220*np.linspace(0,1,signal_len)) + 0.2*np.random.randn(signal_len)
        sp = np.sin(2*np.pi*440*np.linspace(0,1,signal_len)) + 0.2*np.random.randn(signal_len)
    else:                 # Violent
        label = 2
        ax = np.sin(2*np.pi*320*np.linspace(0,1,signal_len)) + 0.4*np.random.randn(signal_len)
        ay = np.sin(2*np.pi*350*np.linspace(0,1,signal_len)) + 0.4*np.random.randn(signal_len)
        az = np.sin(2*np.pi*400*np.linspace(0,1,signal_len)) + 0.4*np.random.randn(signal_len)
        sp = np.sin(2*np.pi*500*np.linspace(0,1,signal_len)) + 0.4*np.random.randn(signal_len)
    signals_train.append((ax, ay, az, sp))
    labels_train.append(label)

# --- Repeat for validation data (smaller, 6/class for testing) ---
signals_val = []
labels_val = []
for i in range(18):
    if i < 6:
        label = 0; freq = 80
    elif i < 12:
        label = 1; freq = 160
    else:
        label = 2; freq = 320

    ax = np.sin(2*np.pi*freq*np.linspace(0,1,signal_len)) + 0.15*np.random.randn(signal_len)
    ay = np.sin(2*np.pi*(freq+40)*np.linspace(0,1,signal_len)) + 0.15*np.random.randn(signal_len)
    az = np.sin(2*np.pi*(freq+80)*np.linspace(0,1,signal_len)) + 0.15*np.random.randn(signal_len)
    sp = np.sin(2*np.pi*(freq+160)*np.linspace(0,1,signal_len)) + 0.15*np.random.randn(signal_len)
    signals_val.append((ax, ay, az, sp))
    labels_val.append(label)


# (B) Build training and validation image datasets
X_train, y_train = build_image_dataset(signals_train, labels_train)
X_val, y_val = build_image_dataset(signals_val, labels_val)

# (C) Build and compile model
model = build_shuffle_bilstm(input_shape=(256,256,3), n_classes=3)
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# (D) Configure callbacks
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('../models/best_shuffle_bilstm.keras', save_best_only=True)
]

# (E) Train!
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=95,
    callbacks=callbacks
)
