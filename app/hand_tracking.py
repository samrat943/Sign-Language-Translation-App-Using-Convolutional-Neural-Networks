import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandDetector:
    """High-Precision Hand Detector using MediaPipe Tasks HandLandmarker."""

    # 21 Hand Landmark Skeletal Connections
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
        (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
        (13, 17), (17, 18), (18, 19), (19, 20),# Pinky finger
        (0, 17)                                # Palm base
    ]

    def __init__(self, max_hands=1, num_hands=1, detection_con=0.5, track_con=0.5, min_detection_confidence=0.5, **kwargs):
        self.num_hands = max_hands if max_hands != 1 else num_hands
        self.min_confidence = min_detection_confidence if min_detection_confidence != 0.5 else detection_con
        self.detector = None
        self._init_landmarker()

    def _init_landmarker(self):
        """Downloads hand_landmarker.task model file if missing and initializes detector."""
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, "hand_landmarker.task")
        model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

        if not os.path.exists(model_path):
            print(f"Downloading MediaPipe HandLandmarker task model -> {model_path}...")
            urllib.request.urlretrieve(model_url, model_path)
            print("Download complete.")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_confidence,
            min_hand_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        print("MediaPipe HandLandmarker initialized successfully.")

    def process_frame(self, frame, draw_landmarks=True, padding=30):
        """
        Processes BGR frame with MediaPipe HandLandmarker.
        Returns:
            annotated_frame (np.ndarray): Frame with hand keypoints and bounding box.
            hand_crop (np.ndarray or None): Cropped hand image.
            bbox (tuple or None): Bounding box coordinates (xmin, ymin, xmax, ymax).
            raw_landmarks (list or None): 21 MediaPipe hand landmarks.
        """
        if frame is None or self.detector is None:
            return frame, None, None, None

        h, w, c = frame.shape
        annotated_frame = frame.copy()
        hand_crop = None
        bbox = None
        raw_landmarks = None

        # Convert BGR to RGB for MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hands
        results = self.detector.detect(mp_image)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                raw_landmarks = hand_landmarks
                # Convert normalized landmarks to pixel coordinates
                landmark_coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

                x_coords = [pt[0] for pt in landmark_coords]
                y_coords = [pt[1] for pt in landmark_coords]

                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)

                # Add padding around bounding box
                xmin = max(0, xmin - padding)
                ymin = max(0, ymin - padding)
                xmax = min(w, xmax + padding)
                ymax = min(h, ymax + padding)

                bbox = (xmin, ymin, xmax, ymax)

                # Crop hand region if box is valid
                if (xmax - xmin) > 15 and (ymax - ymin) > 15:
                    hand_crop = frame[ymin:ymax, xmin:xmax]

                if draw_landmarks:
                    # Draw skeletal connections
                    for start_idx, end_idx in self.HAND_CONNECTIONS:
                        pt1 = landmark_coords[start_idx]
                        pt2 = landmark_coords[end_idx]
                        cv2.line(annotated_frame, pt1, pt2, (0, 255, 128), 2)

                    # Draw 21 landmark dots
                    for cx, cy in landmark_coords:
                        cv2.circle(annotated_frame, (cx, cy), 4, (0, 215, 255), -1)
                        cv2.circle(annotated_frame, (cx, cy), 6, (0, 255, 128), 1)

                    # Draw bounding box rectangle
                    cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 128), 2)
                    cv2.putText(
                        annotated_frame, "HAND DETECTED", (xmin, max(20, ymin - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2
                    )

                break  # Process primary hand

        return annotated_frame, hand_crop, bbox, raw_landmarks
