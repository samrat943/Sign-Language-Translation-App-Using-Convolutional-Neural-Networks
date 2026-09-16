import os
import json
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class LandmarkGestureClassifier:
    """
    High-Accuracy Geometric & Landmark-based Sign Language Classifier.
    Evaluates 21 3D MediaPipe hand joint coordinates and angles.
    Supports user custom gesture memory & instant training.
    """

    DEFAULT_GESTURE_MAP = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
        9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',
        17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
    }

    def __init__(self, data_file=None):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if data_file is None:
            data_file = os.path.join(project_root, "data", "custom_gestures.json")

        self.data_file = data_file
        self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self.is_trained = False
        self.samples = []
        self.load_custom_gestures()

    @staticmethod
    def dist(pt1, pt2):
        """Calculates 2D Euclidean distance between two MediaPipe landmark points."""
        return math.sqrt((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2)

    @staticmethod
    def extract_features(hand_landmarks):
        """Extracts 42 normalized relative landmark coordinates (relative to wrist)."""
        if not hand_landmarks:
            return None

        wrist_x = hand_landmarks[0].x
        wrist_y = hand_landmarks[0].y

        features = []
        for lm in hand_landmarks:
            features.extend([lm.x - wrist_x, lm.y - wrist_y])

        feat_arr = np.array(features, dtype=np.float32)
        max_val = np.max(np.abs(feat_arr))
        if max_val > 0:
            feat_arr /= max_val

        return feat_arr

    def predict_geometric_rules(self, hand_landmarks):
        """
        Rule-based geometric classifier for 21 MediaPipe hand joint landmarks.
        Determines finger extension states and joint distances.
        """
        if not hand_landmarks or len(hand_landmarks) < 21:
            return None, 0.0

        lm = hand_landmarks
        wrist = lm[0]

        # Finger tip landmarks: 4=Thumb, 8=Index, 12=Middle, 16=Ring, 20=Pinky
        # Finger PIP/MCP landmarks: 2,6,10,14,18
        dist_wrist = lambda idx: self.dist(lm[idx], wrist)

        # Check finger extensions relative to wrist and PIP joints
        index_up  = self.dist(lm[8], wrist) > self.dist(lm[6], wrist) and lm[8].y < lm[6].y
        middle_up = self.dist(lm[12], wrist) > self.dist(lm[10], wrist) and lm[12].y < lm[10].y
        ring_up   = self.dist(lm[16], wrist) > self.dist(lm[14], wrist) and lm[16].y < lm[14].y
        pinky_up  = self.dist(lm[20], wrist) > self.dist(lm[18], wrist) and lm[20].y < lm[18].y

        # Thumb extension: tip distance to pinky base (lm 17)
        thumb_out = self.dist(lm[4], lm[17]) > self.dist(lm[2], lm[17])

        # Distances between finger tips
        index_thumb_dist  = self.dist(lm[8], lm[4])
        middle_thumb_dist = self.dist(lm[12], lm[4])
        index_middle_dist = self.dist(lm[8], lm[12])

        # Classify based on finger posture combinations
        # 1. 'Y': Only Thumb and Pinky out
        if thumb_out and pinky_up and not index_up and not middle_up and not ring_up:
            return 'Y', 0.95

        # 2. 'L': Index up and Thumb out, others folded
        if index_up and thumb_out and not middle_up and not ring_up and not pinky_up:
            return 'L', 0.95

        # 3. 'I': Only Pinky up
        if pinky_up and not index_up and not middle_up and not ring_up and not thumb_out:
            return 'I', 0.92

        # 4. 'W': Index, Middle, Ring up, Pinky folded
        if index_up and middle_up and ring_up and not pinky_up:
            return 'W', 0.93

        # 5. 'V' or 'U': Index and Middle up
        if index_up and middle_up and not ring_up and not pinky_up:
            if index_middle_dist > 0.08:
                return 'V', 0.94
            else:
                return 'U', 0.92

        # 6. 'B': All 4 fingers up together
        if index_up and middle_up and ring_up and pinky_up:
            return 'B', 0.96

        # 7. 'F': Index touches Thumb (circle), Middle, Ring, Pinky up
        if index_thumb_dist < 0.06 and middle_up and ring_up and pinky_up:
            return 'F', 0.94

        # 8. 'D': Index up, Thumb touches Middle tip
        if index_up and middle_thumb_dist < 0.08 and not ring_up and not pinky_up:
            return 'D', 0.91

        # 9. 'C' or 'O': All fingers curved
        if not index_up and not middle_up and not ring_up and not pinky_up:
            if index_thumb_dist < 0.06 and middle_thumb_dist < 0.06:
                return 'O', 0.90
            elif thumb_out:
                return 'A', 0.92
            else:
                return 'E', 0.88

        # Default fallback
        if index_up and not middle_up and not ring_up and not pinky_up:
            return 'D', 0.85

        return 'A', 0.75

    def add_sample(self, label, hand_landmarks):
        """Records a new landmark gesture sample into memory and file."""
        feat = self.extract_features(hand_landmarks)
        if feat is not None:
            self.samples.append({
                "label": str(label).upper(),
                "features": feat.tolist()
            })
            self.save_custom_gestures()
            self.train()
            return True
        return False

    def save_custom_gestures(self):
        """Saves custom recorded gesture samples to JSON file."""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        with open(self.data_file, "w", encoding="utf-8") as f:
            json.dump(self.samples, f, indent=2)
        print(f"Saved {len(self.samples)} custom gesture samples to '{self.data_file}'.")

    def load_custom_gestures(self):
        """Loads custom recorded gesture samples from JSON file if available."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    self.samples = json.load(f)
                if self.samples:
                    print(f"Loaded {len(self.samples)} custom gesture samples.")
                    self.train()
            except Exception as e:
                print(f"Error loading custom gestures: {e}")

    def train(self):
        """Trains the Random Forest classifier on recorded gesture samples."""
        if not self.samples:
            self.is_trained = False
            return False

        X = [s["features"] for s in self.samples]
        y = [s["label"] for s in self.samples]

        if len(set(y)) < 1:
            self.is_trained = False
            return False

        self.classifier.fit(X, y)
        self.is_trained = True
        return True

    def predict(self, hand_landmarks):
        """
        Predicts gesture letter sign and confidence.
        Uses Custom Trained ML model if samples exist, otherwise geometric landmark rules.
        """
        if hand_landmarks is None:
            return None, 0.0

        # 1. Use Custom Trained Model if user recorded samples
        if self.is_trained:
            feat = self.extract_features(hand_landmarks)
            if feat is not None:
                probs = self.classifier.predict_proba([feat])[0]
                max_idx = np.argmax(probs)
                confidence = float(probs[max_idx])
                predicted_letter = str(self.classifier.classes_[max_idx])
                return predicted_letter, confidence

        # 2. Otherwise, use robust geometric landmark rules
        return self.predict_geometric_rules(hand_landmarks)

    def clear_samples(self):
        """Resets all recorded gesture samples."""
        self.samples = []
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        self.is_trained = False
