from tensorflow.keras.models import load_model
from dataset_builder import build_image_dataset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Load test data

# --- Simulate test dataset: 3 samples/class (for testing) ---
signal_len = 4096
signals_test = []
labels_test = []

for i in range(9):  # 3 stable, 3 transition, 3 violent
    if i < 3:            # Stable
        label = 0
        ax = np.sin(2*np.pi*80*np.linspace(0,1,signal_len)) + 0.1*np.random.randn(signal_len)
        ay = np.sin(2*np.pi*120*np.linspace(0,1,signal_len)) + 0.1*np.random.randn(signal_len)
        az = np.sin(2*np.pi*200*np.linspace(0,1,signal_len)) + 0.1*np.random.randn(signal_len)
        sp = np.sin(2*np.pi*340*np.linspace(0,1,signal_len)) + 0.1*np.random.randn(signal_len)
    elif i < 6:          # Transition
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
    
    signals_test.append((ax, ay, az, sp))
    labels_test.append(label)

X_test, y_test = build_image_dataset(signals_test, labels_test)
model = load_model('../models/best_shuffle_bilstm.keras')
y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=['Stable','Transition','Violent']))
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.title('Confusion Matrix')
plt.savefig('../results/confusion_matrix.png')
plt.show()
import matplotlib.pyplot as plt

# After generating X_train in train.py or eval.py
# Find one sample from each class
class_indices = [np.where(y_test == i)[0][0] for i in range(3)]

plt.figure(figsize=(15,4))
class_names = ['Stable', 'Transition', 'Violent']
for idx, (i, name) in enumerate(zip(class_indices, class_names)):
    plt.subplot(1,3,idx+1)
    plt.imshow(X_test[i])
    plt.title(f"{name} (Class {y_test[i]})")
    plt.axis('off')
plt.suptitle('SPWVD Time-Frequency Images for Each Vibration State')
plt.tight_layout()
plt.savefig('../results/all_classes_images.png', dpi=150)
plt.show()
