import os
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# 🤟 Sign Language Translator: Data Exploration & Preprocessing

## Overview
This notebook performs exploratory data analysis (EDA) and preprocessing for the **Sign Language MNIST** dataset.
The Sign Language MNIST dataset represents American Sign Language (ASL) alphabetic gestures for letters **A–I** and **K–Y** (24 classes total; **J** and **Z** are excluded as they involve motion gestures).

Each image is a **28x28 grayscale** image represented by 784 pixel values (0–255).

### Notebook Objectives:
1. **Dataset Acquisition & Loading**: Provide robust manual download instructions and automated setup logic.
2. **Exploratory Data Analysis (EDA)**: Inspect class distributions, image dimensions, and visual samples per letter.
3. **Data Augmentation**: Demonstrate realistic augmentation techniques tailored for hand gesture recognition.
4. **Data Preprocessing & Export**: Normalize images, perform stratified train/val/test splitting, and export clean arrays to `data/processed/`.
"""))

# Cell 2: Imports
cells.append(nbf.v4.new_code_cell("""import os
import json
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams["font.size"] = 10

print(f"TensorFlow version: {tf.__version__}")
print(f"OpenCV version: {cv2.__version__}")
"""))

# Cell 3: Dataset Acquisition Markdown
cells.append(nbf.v4.new_markdown_cell("""## 1. Dataset Acquisition & Loading

### Manual Download Instructions (Kaggle)
If you prefer to download the full Sign Language MNIST dataset directly from Kaggle:
1. Visit [Sign Language MNIST on Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist).
2. Download `sign_mnist_train.csv` and `sign_mnist_test.csv`.
3. Place both CSV files inside `data/raw/` relative to the project root.

### Automated Setup & Fallback Generator
The cell below checks if `data/raw/sign_mnist_train.csv` exists. If not found locally, it creates realistic sample data so that this notebook remains 100% executable out of the box.
"""))

# Cell 4: Data Loader Code
cells.append(nbf.v4.new_code_cell("""# Define directories
RAW_DATA_DIR = os.path.join("..", "data", "raw")
PROCESSED_DATA_DIR = os.path.join("..", "data", "processed")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

train_csv_path = os.path.join(RAW_DATA_DIR, "sign_mnist_train.csv")
test_csv_path = os.path.join(RAW_DATA_DIR, "sign_mnist_test.csv")

# Class mapping: 0-24 maps to A-Z (excluding J=9 and Z=25)
LABEL_MAP = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

def generate_synthetic_sign_mnist(filename, num_samples=1200):
    # Generates synthetic Sign Language MNIST CSV data if raw dataset is absent.
    valid_labels = list(LABEL_MAP.keys())
    labels = np.random.choice(valid_labels, size=num_samples)
    
    # Generate structured noise resembling hand contours
    pixels = []
    for lbl in labels:
        # Create a simple shape pattern based on label
        img = np.full((28, 28), 50, dtype=np.uint8)
        cx, cy = 14 + (lbl % 5) - 2, 14 + (lbl // 5) - 2
        cv2.circle(img, (cx, cy), 8, 200, -1)
        cv2.GaussianBlur(img, (3, 3), 0, dst=img)
        pixels.append(img.flatten())
        
    pixels = np.array(pixels)
    cols = ['label'] + [f'pixel{i+1}' for i in range(784)]
    df = pd.DataFrame(np.column_stack([labels, pixels]), columns=cols)
    df.to_csv(filename, index=False)
    print(f"Generated synthetic dataset with {num_samples} rows -> {filename}")

if not os.path.exists(train_csv_path):
    print("Raw train dataset not found. Generating sample data...")
    generate_synthetic_sign_mnist(train_csv_path, num_samples=2400)

if not os.path.exists(test_csv_path):
    print("Raw test dataset not found. Generating sample data...")
    generate_synthetic_sign_mnist(test_csv_path, num_samples=600)

# Load data into pandas DataFrames
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

print(f"Loaded train dataset: {train_df.shape}")
print(f"Loaded test dataset: {test_df.shape}")
"""))

# Cell 5: Data Inspection Markdown
cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset Inspection & Quality Check

We inspect missing values, data types, and pixel intensity ranges across features.
"""))

# Cell 6: Data Inspection Code
cells.append(nbf.v4.new_code_cell("""# Check null values
null_count_train = train_df.isnull().sum().sum()
null_count_test = test_df.isnull().sum().sum()

print(f"Missing values in train set: {null_count_train}")
print(f"Missing values in test set:  {null_count_test}")

# Extract labels and pixel arrays
y_raw = train_df['label'].values
x_raw = train_df.drop('label', axis=1).values

print(f"Pixel value min: {x_raw.min()}, max: {x_raw.max()}")
print(f"Number of unique classes present: {len(np.unique(y_raw))}")
"""))

# Cell 7: EDA & Class Distribution Markdown
cells.append(nbf.v4.new_markdown_cell("""## 3. Exploratory Data Analysis (EDA)

### Class Distribution
We analyze the distribution of instances across all 24 sign language gesture categories to verify whether class imbalance exists.
"""))

# Cell 8: Class Distribution Plot Code
cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(14, 5))
class_counts = train_df['label'].value_counts().sort_index()
letters = [LABEL_MAP.get(lbl, str(lbl)) for lbl in class_counts.index]

bars = plt.bar(letters, class_counts.values, color=sns.color_palette("mako", len(letters)))
plt.title("Sign Language Class Distribution (Training Set)", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("ASL Letter Gesture", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)

# Annotate counts on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + (0.01 * max(class_counts.values)), 
             int(yval), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# Print statistical summary of distribution
print(f"Mean samples per class: {class_counts.mean():.1f}")
print(f"Min samples per class:  {class_counts.min()} (Class {LABEL_MAP.get(class_counts.idxmin())})")
print(f"Max samples per class:  {class_counts.max()} (Class {LABEL_MAP.get(class_counts.idxmax())})")
"""))

# Cell 9: Sample Visualization Markdown
cells.append(nbf.v4.new_markdown_cell("""### Visualizing Sample Images per ASL Gesture
We display a sample image for each letter in the dataset to inspect image quality, contrast, and visual patterns.
"""))

# Cell 10: Sample Visualization Grid Code
cells.append(nbf.v4.new_code_cell("""# Reshape pixel vector to 28x28 images
images_2d = x_raw.reshape(-1, 28, 28)

fig, axes = plt.subplots(4, 6, figsize=(15, 10))
fig.suptitle("Sample ASL Hand Gestures per Letter Class", fontsize=16, fontweight="bold", y=0.98)

unique_labels = sorted(list(LABEL_MAP.keys()))

for idx, label_val in enumerate(unique_labels):
    row, col = idx // 6, idx % 6
    ax = axes[row, col]
    
    # Find first instance of this label
    sample_idx = np.where(y_raw == label_val)[0][0]
    sample_img = images_2d[sample_idx]
    letter = LABEL_MAP[label_val]
    
    ax.imshow(sample_img, cmap='gray')
    ax.set_title(f"Class {label_val}: '{letter}'", fontsize=11, fontweight="semibold")
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
"""))

# Cell 11: Data Augmentation Markdown
cells.append(nbf.v4.new_markdown_cell("""## 4. Data Augmentation Visualizations

Data augmentation improves model robustness against real-world camera variations (slight hand rotation, lighting changes, distance zoom).

> **Note on Flips**: For sign language gestures, horizontal flips mirror hand orientation (e.g., left hand vs. right hand). While valid for general hand recognition, vertical flips are avoided as upside-down signs alter gesture meanings.
"""))

# Cell 12: Data Augmentation Visualization Code
cells.append(nbf.v4.new_code_cell("""def apply_augmentations(img):
    # Applies rotation, brightness shift, and zoom to a 28x28 grayscale image.
    h, w = img.shape[:2]
    
    # 1. Rotation (+-15 degrees)
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # 2. Brightness adjustment
    brightness_factor = np.random.uniform(0.7, 1.3)
    brightened = np.clip(img * brightness_factor, 0, 255).astype(np.uint8)
    
    # 3. Zoom / Shift
    zoom_factor = np.random.uniform(0.9, 1.1)
    M_zoom = cv2.getRotationMatrix2D((w//2, h//2), 0, zoom_factor)
    zoomed = cv2.warpAffine(img, M_zoom, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # 4. Combined
    M_comb = cv2.getRotationMatrix2D((w//2, h//2), angle, zoom_factor)
    combined = cv2.warpAffine(brightened, M_comb, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return rotated, brightened, zoomed, combined

# Pick a sample image to demonstrate augmentation
sample_img = images_2d[0].astype(np.uint8)
rot, bright, zoom, comb = apply_augmentations(sample_img)

fig, axes = plt.subplots(1, 5, figsize=(16, 4))
titles = ["Original", "Rotated (±15°)", "Brightness", "Zoomed (±10%)", "Combined Augmentation"]
imgs = [sample_img, rot, bright, zoom, comb]

for ax, title, image in zip(axes, titles, imgs):
    ax.imshow(image, cmap='gray')
    ax.set_title(title, fontsize=11, fontweight="semibold")
    ax.axis('off')

plt.tight_layout()
plt.show()
"""))

# Cell 13: Normalization & Train/Val/Test Split Markdown
cells.append(nbf.v4.new_markdown_cell("""## 5. Normalization, Dataset Splitting & Export

We format the dataset for deep learning models:
1. **Reshape & Normalize**: Reshape pixel vectors into `(N, 28, 28, 1)` and normalize values to `[0, 1]`.
2. **Label Encoding**: Remap non-contiguous class IDs into zero-indexed integers `0–23`.
3. **Stratified Split**:
   - Training Set: 70%
   - Validation Set: 15%
   - Test Set: 15%
4. **Export**: Save preprocessed numpy arrays to `data/processed/` for model training.
"""))

# Cell 14: Splitting & Export Code
cells.append(nbf.v4.new_code_cell("""# Combine raw train and test into one pool for clean stratified split
x_all = np.vstack([x_raw, test_df.drop('label', axis=1).values])
y_all = np.concatenate([y_raw, test_df['label'].values])

# Remap non-contiguous labels (0..24 with 9 missing) to 0..23 range
unique_labels_sorted = sorted(np.unique(y_all))
label_to_index = {orig_lbl: idx for idx, orig_lbl in enumerate(unique_labels_sorted)}
index_to_letter = {idx: LABEL_MAP[orig_lbl] for idx, orig_lbl in enumerate(unique_labels_sorted)}

y_encoded = np.array([label_to_index[lbl] for lbl in y_all])

# Reshape & Normalize images: (N, 28, 28, 1), float32 in [0, 1]
x_all_norm = (x_all.reshape(-1, 28, 28, 1)).astype(np.float32) / 255.0

# 1st Split: 70% Train, 30% Temp (Val + Test)
x_train, x_temp, y_train, y_temp = train_test_split(
    x_all_norm, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
)

# 2nd Split: Split Temp into 50% Val, 50% Test (15% overall each)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Dataset Shapes & Splits:")
print(f" - Train Set:      x={x_train.shape}, y={y_train.shape}")
print(f" - Validation Set: x={x_val.shape}, y={y_val.shape}")
print(f" - Test Set:       x={x_test.shape}, y={y_test.shape}")
print(f" - Total Classes:  {len(index_to_letter)}")

# Save processed arrays
np.save(os.path.join(PROCESSED_DATA_DIR, "train_x.npy"), x_train)
np.save(os.path.join(PROCESSED_DATA_DIR, "train_y.npy"), y_train)
np.save(os.path.join(PROCESSED_DATA_DIR, "val_x.npy"), x_val)
np.save(os.path.join(PROCESSED_DATA_DIR, "val_y.npy"), y_val)
np.save(os.path.join(PROCESSED_DATA_DIR, "test_x.npy"), x_test)
np.save(os.path.join(PROCESSED_DATA_DIR, "test_y.npy"), y_test)

# Save label mapping metadata
with open(os.path.join(PROCESSED_DATA_DIR, "label_map.json"), "w") as f:
    json.dump(index_to_letter, f, indent=2)

print(f"Preprocessed arrays & label map successfully saved to '{PROCESSED_DATA_DIR}'!")
"""))

# Cell 15: Summary Markdown
cells.append(nbf.v4.new_markdown_cell("""## 6. Summary & Key Findings

### Data Analysis Key Findings
- **Dataset Structure**: The dataset contains 28x28 pixel grayscale images representing 24 ASL alphabetic hand gestures.
- **Class Balance**: All 24 gesture classes are cleanly represented with minimal class imbalance across training and test splits.
- **Preprocessing Pipeline**:
  - Pixel values normalized to `[0.0, 1.0]`.
  - Reshaped to tensor format `(N, 28, 28, 1)`.
  - Labels zero-indexed (`0–23`).
  - Saved to `data/processed/` with a 70/15/15 stratified train/val/test split.

### Insights & Next Steps
- Proceed to **`02_model_training.ipynb`** to construct and train a Convolutional Neural Network (CNN) using TensorFlow/Keras.
"""))

nb['cells'] = cells

notebook_path = os.path.join("notebooks", "01_data_exploration.ipynb")
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully created notebook at {notebook_path}")
