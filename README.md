# Sign Language Translator

A real-time American Sign Language (ASL) translation web application powered by MediaPipe 3D Hand Landmark Tracking, TensorFlow Lite deep learning, Scikit-Learn gesture memory, and a Streamlit web interface.

---

## Key Features

- **Real-Time Video Translation**: Translates hand signs from live webcam video streams with ultra-low latency.
- **MediaPipe 3D Landmark Tracking**: Tracks 21 anatomical hand keypoints, making detection invariant to background lighting, skin tone, or camera distance.
- **Gesture Calibration & Memory**: Teach the AI custom hand signs directly from the UI. Recorded gestures are saved and retrained in real time.
- **Sentence Builder**: Accumulates recognized gesture signs into full sentences with consecutive frame stability checks to eliminate flicker.
- **Session Controls & Export**: Interactive controls for Space, Backspace, and Clear, with downloadable timestamped CSV session logs.
- **Performance Monitor**: Integrated real-time FPS counter and frame processing pipeline.

---

## Project Structure

```
sign-language-translator/
├── app/
│   ├── main.py                     # Streamlit application entry point
│   ├── hand_tracking.py            # MediaPipe 3D landmark detection module
│   ├── inference.py                # TensorFlow Lite model inference engine
│   ├── landmark_classifier.py      # Custom gesture memory & keypoint classifier
│   ├── asl_chart.png               # ASL gesture reference chart
│   └── app.py                      # Application launcher alias
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Data exploration, augmentation, & preprocessing
│   └── 02_train_cnn_model.ipynb    # CNN architecture training, evaluation, & TFLite export
├── data/                           # Datasets (gitignored)
│   ├── raw/                        # Raw dataset CSV files
│   ├── processed/                  # Preprocessed train/val/test split numpy arrays
│   └── custom_gestures.json        # Recorded custom gesture landmark samples
├── models/                         # Saved models (gitignored)
│   ├── sign_cnn.h5                 # Keras HDF5 model weights
│   └── sign_cnn.tflite             # Quantized TFLite model for real-time inference
├── start.bat                       # One-click Windows startup script
├── venv/                           # Python virtual environment
├── requirements.txt                # Dependency specifications
└── README.md                       # Project documentation
```

---

## Quick Start (Windows)

Simply double-click **`start.bat`** or run:

```cmd
start.bat
```

---

## Setup & Manual Installation

### 1. Environment Setup

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application Manually

Launch the Streamlit web application:

```powershell
streamlit run app/main.py
```

1. Open your web browser at `http://localhost:8501`.
2. Select your webcam index and click **Start Camera**.
3. Hold your hand in camera view to translate ASL gestures in real time.
4. Expand **Gesture Calibration & Memory** to record and train custom hand signs.

---

## Model Training Workflow

To retrain the CNN models on updated or custom datasets:

1. **Launch JupyterLab**:
   ```powershell
   jupyter lab
   ```
2. **Data Exploration & Preprocessing**:
   - Run `notebooks/01_data_exploration.ipynb` to process datasets and save split arrays to `data/processed/`.
3. **Model Training & TFLite Export**:
   - Run `notebooks/02_train_cnn_model.ipynb` to train the CNN model and automatically export `models/sign_cnn.h5` and `models/sign_cnn.tflite`.

### Notebooks on Google Colab

- [Data Exploration & Preprocessing](https://colab.research.google.com/drive/1fEOOhLk_1XU6ftWQA652mFHEifZBY5FT?usp=sharing)
- [CNN Model Training & TFLite Export](https://colab.research.google.com/drive/1xjDDYRZf30x13hqnjBA1FZY0phPprFR4?usp=sharing)

---

## Tech Stack

- **Deep Learning & Inference**: TensorFlow, Keras, TensorFlow Lite
- **Computer Vision & Tracking**: OpenCV, MediaPipe Tasks
- **Machine Learning**: Scikit-Learn
- **Data & Analytics**: NumPy, Pandas, Matplotlib, Seaborn
- **Web Interface**: Streamlit
