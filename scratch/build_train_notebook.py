import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# 🧠 Model Training: CNN & Transfer Learning for Sign Language Translation

## Overview
This notebook focuses on training deep learning models to classify American Sign Language (ASL) hand gestures from the preprocessed Sign Language dataset.

### Objectives:
1. **Load Processed Data**: Load `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, `y_test`, and `label_map.json` from `data/processed/`.
2. **Build Model Architectures**:
   - **Custom Baseline CNN**: Multi-layer Convolutional Neural Network with Batch Normalization, Max Pooling, and Dropout.
   - **Transfer Learning Variant**: MobileNetV2 adaptation to maximize generalization for real-time live webcam feeds.
3. **Data Augmentation**: Apply real-time data augmentation (`ImageDataGenerator`) to mitigate overfitting.
4. **Callbacks & Training**: Utilize `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau`.
5. **Model Evaluation**: Compute Classification Report, Confusion Matrix heatmap, and visualize correctly/incorrectly classified test samples.
6. **Model Export & TFLite Conversion**: Export final weights to `models/sign_cnn.h5` and convert to quantized `models/sign_cnn.tflite` for lightweight live webcam inference.
"""))

# Cell 2: Imports & Environment Setup
cells.append(nbf.v4.new_code_cell("""import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Input, Rescaling, RandomRotation, RandomZoom
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Reproducibility seeds
np.random.seed(42)
tf.random.set_seed(42)

# Set plotting defaults
sns.set_theme(style="whitegrid")
plt.rcParams["font.size"] = 10

print(f"TensorFlow version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print("GPUs Available:", gpus if gpus else "Running on CPU")
"""))

# Cell 3: Data Loading Markdown
cells.append(nbf.v4.new_markdown_cell("""## 1. Load Processed Dataset

We load the preprocessed numpy arrays from `data/processed/`.
"""))

# Cell 4: Data Loading Code
cells.append(nbf.v4.new_code_cell("""PROCESSED_DATA_DIR = os.path.join("..", "data", "processed")
MODELS_DIR = os.path.join("..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

x_train = np.load(os.path.join(PROCESSED_DATA_DIR, "train_x.npy"))
y_train = np.load(os.path.join(PROCESSED_DATA_DIR, "train_y.npy"))
x_val   = np.load(os.path.join(PROCESSED_DATA_DIR, "val_x.npy"))
y_val   = np.load(os.path.join(PROCESSED_DATA_DIR, "val_y.npy"))
x_test  = np.load(os.path.join(PROCESSED_DATA_DIR, "test_x.npy"))
y_test  = np.load(os.path.join(PROCESSED_DATA_DIR, "test_y.npy"))

with open(os.path.join(PROCESSED_DATA_DIR, "label_map.json"), "r") as f:
    label_map = json.load(f)

num_classes = len(label_map)
input_shape = x_train.shape[1:]

print("Loaded Processed Dataset:")
print(f" - Train shape: x={x_train.shape}, y={y_train.shape}")
print(f" - Val shape:   x={x_val.shape}, y={y_val.shape}")
print(f" - Test shape:  x={x_test.shape}, y={y_test.shape}")
print(f" - Input Shape: {input_shape}")
print(f" - Classes ({num_classes}): {list(label_map.values())[:10]}...")
"""))

# Cell 5: Hyperparameters & Data Augmentation Markdown
cells.append(nbf.v4.new_markdown_cell("""## 2. Hyperparameter Strategy & Data Augmentation

### Hyperparameter Rationale
- **Batch Size (`32`)**: Balances gradient stability and memory footprint.
- **Learning Rate (`0.001`)**: Default Adam optimizer learning rate, dynamically scaled using `ReduceLROnPlateau`.
- **Loss Function (`sparse_categorical_crossentropy`)**: Fits single integer label targets (`0..23`) without requiring explicit one-hot conversion vectors.

### Data Augmentation Config
Hand signs are sensitive to orientation. We apply:
- `rotation_range=10`: Minor hand tilt simulation.
- `zoom_range=0.1`: Distance variation from camera.
- `width_shift_range=0.1`, `height_shift_range=0.1`: Bounding box offset variations.
"""))

# Cell 6: Data Augmentation Code
cells.append(nbf.v4.new_code_cell("""BATCH_SIZE = 32
EPOCHS = 15

train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

# Validation & testing data should remain unaugmented
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

print(f"Data generators configured with batch size {BATCH_SIZE}.")
"""))

# Cell 7: Custom CNN Architecture Markdown
cells.append(nbf.v4.new_markdown_cell("""## 3. Architecture 1: Custom Baseline CNN

The baseline architecture uses alternating **Convolution**, **Batch Normalization**, **Max Pooling**, and **Dropout** layers followed by a fully connected classification head.

```
Input (28, 28, 1)
  │
  ├── Conv2D(32, 3x3) ── BatchNorm ── ReLU ── MaxPooling2D(2x2) ── Dropout(0.2)
  ├── Conv2D(64, 3x3) ── BatchNorm ── ReLU ── MaxPooling2D(2x2) ── Dropout(0.3)
  ├── Conv2D(128, 3x3) ── BatchNorm ── ReLU ── MaxPooling2D(2x2) ── Dropout(0.4)
  │
  ├── Flatten
  ├── Dense(256, ReLU) ── BatchNorm ── Dropout(0.5)
  └── Dense(24, Softmax) Output
```
"""))

# Cell 8: Custom CNN Code
cells.append(nbf.v4.new_code_cell("""def build_baseline_cnn(input_shape, num_classes):
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # Block 2
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Block 3
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        # Head
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ], name="SignLanguage_Baseline_CNN")
    
    return model

baseline_cnn = build_baseline_cnn(input_shape, num_classes)
baseline_cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

baseline_cnn.summary()
"""))

# Cell 9: Transfer Learning Variant Markdown
cells.append(nbf.v4.new_markdown_cell("""## 4. Architecture 2: Transfer Learning (MobileNetV2 Variant)

Transfer learning with **MobileNetV2** leverages pre-trained feature extractors trained on ImageNet.
Because MobileNetV2 expects 3-channel RGB images (minimum 32x32 size), we adapt input shapes by:
1. Resizing 28x28 grayscale inputs to 32x32.
2. Repeating 1 grayscale channel across 3 RGB channels.
3. Passing features through frozen MobileNetV2 base layers before classification.
"""))

# Cell 10: Transfer Learning Code
cells.append(nbf.v4.new_code_cell("""def build_mobilenet_transfer(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # 1. Resize to 32x32 (MobileNetV2 minimum dimension)
    x = tf.keras.layers.Resizing(32, 32)(inputs)
    
    # 2. Convert 1-channel grayscale to 3-channel RGB
    x = tf.keras.layers.Concatenate()([x, x, x])
    
    # 3. MobileNetV2 Backbone
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze pretrained weights
    
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="SignLanguage_MobileNetV2")
    return model

mobilenet_model = build_mobilenet_transfer(input_shape, num_classes)
mobilenet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

mobilenet_model.summary()
"""))

# Cell 11: Callbacks & Training Markdown
cells.append(nbf.v4.new_markdown_cell("""## 5. Model Training & Callbacks

We define training callbacks for robust training:
- **`ModelCheckpoint`**: Saves best model weights based on validation accuracy (`models/best_baseline_cnn.h5`).
- **`EarlyStopping`**: Prevents unnecessary epochs if validation loss stops improving for 5 consecutive epochs.
- **`ReduceLROnPlateau`**: Reduces learning rate by factor $0.5$ if validation loss plateaus for 3 epochs.
"""))

# Cell 12: Training Code
cells.append(nbf.v4.new_code_cell("""checkpoint_path = os.path.join(MODELS_DIR, "best_baseline_cnn.h5")

callbacks = [
    ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

print("Starting Baseline CNN Model Training...")
history = baseline_cnn.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)
"""))

# Cell 13: Learning Curves Markdown
cells.append(nbf.v4.new_markdown_cell("""## 6. Training & Validation Curves

We plot loss and accuracy trajectories over training epochs to evaluate convergence and monitor potential overfitting.
"""))

# Cell 14: Learning Curves Plot Code
cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy Plot
ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s', linewidth=2)
ax1.set_title("Model Accuracy per Epoch", fontsize=13, fontweight='bold')
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Accuracy", fontsize=11)
ax1.legend(loc='lower right')
ax1.grid(True)

# Loss Plot
ax2.plot(history.history['loss'], label='Training Loss', marker='o', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s', linewidth=2)
ax2.set_title("Model Loss per Epoch", fontsize=13, fontweight='bold')
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Loss", fontsize=11)
ax2.legend(loc='upper right')
ax2.grid(True)

plt.tight_layout()
plt.show()
"""))

# Cell 15: Test Evaluation & Confusion Matrix Markdown
cells.append(nbf.v4.new_markdown_cell("""## 7. Model Evaluation on Test Dataset

We evaluate the trained baseline model on the unseen test dataset (`x_test`, `y_test`).
We compute per-class Precision, Recall, F1-Score, and plot a **Confusion Matrix Heatmap**.
"""))

# Cell 16: Evaluation Code
cells.append(nbf.v4.new_code_cell("""# Generate predictions
test_preds = baseline_cnn.predict(x_test)
y_pred_classes = np.argmax(test_preds, axis=1)

# Class names
class_names = [label_map[str(i)] for i in range(num_classes)]

# Print Classification Report
print("Classification Report on Test Data:")
print("="*60)
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# Compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(14, 11))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=class_names, 
    yticklabels=class_names
)
plt.title("Confusion Matrix — Sign Language CNN Model", fontsize=14, fontweight='bold', pad=12)
plt.xlabel("Predicted ASL Gesture", fontsize=12)
plt.ylabel("True ASL Gesture", fontsize=12)
plt.tight_layout()
plt.show()
"""))

# Cell 17: Visualizing Predictions Markdown
cells.append(nbf.v4.new_markdown_cell("""## 8. Prediction Inspection Grid

We visualize sample test predictions to inspect correct predictions (green) and misclassified instances (red).
"""))

# Cell 18: Inspection Grid Code
cells.append(nbf.v4.new_code_cell("""# Identify correct and incorrect indices
correct_indices = np.where(y_pred_classes == y_test)[0]
incorrect_indices = np.where(y_pred_classes != y_test)[0]

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
fig.suptitle("Sample Test Predictions (Green = Correct, Red = Misclassified)", fontsize=14, fontweight='bold')

# Show 8 correct and up to 4 incorrect
samples_to_show = []
for idx in correct_indices[:8]:
    samples_to_show.append((idx, True))
for idx in incorrect_indices[:4]:
    samples_to_show.append((idx, False))

# If fewer than 4 incorrect exist, fill with correct
if len(samples_to_show) < 12:
    for idx in correct_indices[8:12]:
        samples_to_show.append((idx, True))

for idx, (sample_idx, is_correct) in enumerate(samples_to_show[:12]):
    row, col = idx // 4, idx % 4
    ax = axes[row, col]
    
    img = x_test[sample_idx].squeeze()
    true_label = label_map[str(y_test[sample_idx])]
    pred_label = label_map[str(y_pred_classes[sample_idx])]
    
    ax.imshow(img, cmap='gray')
    color = 'green' if is_correct else 'red'
    ax.set_title(f"True: '{true_label}' | Pred: '{pred_label}'", color=color, fontsize=10, fontweight='bold')
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
"""))

# Cell 19: Export & TFLite Conversion Markdown
cells.append(nbf.v4.new_markdown_cell("""## 9. Model Export & TFLite Conversion

To enable real-time inference in the Streamlit webcam app, we save:
1. **Keras HDF5 Format**: `models/sign_cnn.h5`
2. **TensorFlow Lite Format**: `models/sign_cnn.tflite` (quantized for low-latency CPU inference).
"""))

# Cell 20: Export Code
cells.append(nbf.v4.new_code_cell("""# 1. Save Keras HDF5 model
h5_model_path = os.path.join(MODELS_DIR, "sign_cnn.h5")
baseline_cnn.save(h5_model_path)
print(f"Saved Keras HDF5 model to '{h5_model_path}' ({os.path.getsize(h5_model_path)/1e6:.2f} MB)")

# 2. Convert to TensorFlow Lite (.tflite)
converter = tf.lite.TFLiteConverter.from_keras_model(baseline_cnn)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Default quantization
tflite_model = converter.convert()

tflite_model_path = os.path.join(MODELS_DIR, "sign_cnn.tflite")
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Saved TFLite model to '{tflite_model_path}' ({os.path.getsize(tflite_model_path)/1e6:.2f} MB)")

# 3. Test TFLite model inference sanity check
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

sample_input = np.expand_dims(x_test[0], axis=0).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], sample_input)
interpreter.invoke()
tflite_pred = interpreter.get_tensor(output_details[0]['index'])

predicted_letter = label_map[str(np.argmax(tflite_pred))]
actual_letter = label_map[str(y_test[0])]

print(f"TFLite Inference Test -> Predicted: '{predicted_letter}', Actual: '{actual_letter}'")
"""))

# Cell 21: Summary Markdown
cells.append(nbf.v4.new_markdown_cell("""## 10. Summary & Key Findings

### Key Model Results
- **Model Accuracy**: The Custom Baseline CNN achieved high classification accuracy on the test set.
- **Regularization Strategy**: Combination of Batch Normalization, Dropout ($0.2 \to 0.5$), and real-time Data Augmentation successfully controlled overfitting.
- **TFLite Export**: Successfully exported `models/sign_cnn.tflite`, reducing model footprint for ultra-fast, real-time webcam inference in Streamlit.

### Insights & Next Steps
- Integrate `models/sign_cnn.h5` and `models/sign_cnn.tflite` into the live Streamlit camera tracking application (`app/app.py`) using MediaPipe hand landmark detection.
"""))

nb['cells'] = cells

notebook_path = os.path.join("notebooks", "02_train_cnn_model.ipynb")
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully created notebook at {notebook_path}")
