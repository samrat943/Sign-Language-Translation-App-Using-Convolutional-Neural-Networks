import os
import json
import numpy as np
import cv2
import tensorflow as tf

class SignLanguageClassifier:
    """Inference engine for Sign Language Gesture Classification using TFLite."""

    DEFAULT_LABEL_MAP = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
        9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',
        17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
    }

    def __init__(self, model_path=None, label_map_path=None):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        if model_path is None:
            model_path = os.path.join(project_root, "models", "sign_cnn.tflite")
        if label_map_path is None:
            label_map_path = os.path.join(project_root, "data", "processed", "label_map.json")

        self.model_path = model_path
        self.label_map = self._load_label_map(label_map_path)
        
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._load_tflite_model()

    def _load_label_map(self, path):
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    raw_map = json.load(f)
                    return {int(k): str(v) for k, v in raw_map.items()}
            except Exception as e:
                print(f"Warning: Could not load label map from {path}: {e}")
        return self.DEFAULT_LABEL_MAP

    def _load_tflite_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"TFLite model not found at '{self.model_path}'. Please run training first.")

        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"Loaded TFLite model from '{self.model_path}'.")

    def preprocess_crop(self, hand_crop):
        """
        Preprocesses BGR hand crop into (1, 28, 28, 1) float32 tensor normalized to [0, 1].
        """
        if hand_crop is None or hand_crop.size == 0:
            return None

        # Convert to Grayscale
        gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
        
        # Resize to 28x28
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values to [0.0, 1.0]
        normalized = resized.astype(np.float32) / 255.0
        
        # Reshape to (1, 28, 28, 1)
        tensor = np.expand_dims(normalized, axis=(0, -1))
        return tensor

    def predict(self, hand_crop):
        """
        Runs inference on hand crop.
        Returns:
            predicted_letter (str or None)
            confidence (float): Value between 0.0 and 1.0
            top_probabilities (dict): Top predicted classes and their confidence
        """
        tensor = self.preprocess_crop(hand_crop)
        if tensor is None:
            return None, 0.0, {}

        # Set tensor & invoke
        self.interpreter.set_tensor(self.input_details[0]['index'], tensor)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        pred_class_idx = int(np.argmax(probs))
        confidence = float(probs[pred_class_idx])
        predicted_letter = self.label_map.get(pred_class_idx, "?")

        # Top 3 probabilities
        top_indices = np.argsort(probs)[::-1][:3]
        top_probs = {self.label_map.get(idx, "?"): float(probs[idx]) for idx in top_indices}

        return predicted_letter, confidence, top_probs
