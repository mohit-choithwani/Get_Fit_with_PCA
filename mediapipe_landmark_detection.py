import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import os

class PoseLandmarkExtractor:
    def __init__(self, model_path='pose_landmarker_heavy.task'):
        """
        Initializes the PoseLandmarkExtractor with the specified model.
        """
        self.base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.PoseLandmarkerOptions(
            base_options=self.base_options,
            output_segmentation_masks=True
        )
        self.detector = vision.PoseLandmarker.create_from_options(self.options)
        self.drawing_utils = mp.solutions.drawing_utils
        self.pose_connections = mp.solutions.pose.POSE_CONNECTIONS
        self.landmark_style = mp.solutions.drawing_styles.get_default_pose_landmarks_style()

    def detect_landmarks(self, image_path):
        """
        Detects pose landmarks in the given image and keeps specific indices.

        Args:
            image_path (str): Path to the input image.

        Returns:
            dict: A dictionary containing the selected pose landmarks (x, y) and the annotated image.
        """
        # Indices of landmarks to keep
        selected_indices = [0, 11, 12, 13, 14, 15, 16, 24, 23, 25, 26, 27, 28, 31, 32]
        
        # Load the image
        mp_image = mp.Image.create_from_file(image_path)
        
        # Detect pose landmarks
        detection_result = self.detector.detect(mp_image)
        
        # Extract pose landmarks
        landmarks = detection_result.pose_landmarks
        if landmarks:
            # Flatten landmarks to (x, y) for selected indices
            selected_landmarks = [
                (landmarks[0][index].x, landmarks[0][index].y) 
                for index in selected_indices
            ]
        else:
            selected_landmarks = []
        
        # Annotate image with selected landmarks
        annotated_image = self._draw_landmarks_on_image(
            mp_image.numpy_view(), detection_result, selected_indices
        )
        print(f"selected landmarks : {selected_landmarks}")
        return {
            "selected_landmarks": selected_landmarks,
            "annotated_image": annotated_image
        }

    def _draw_landmarks_on_image(self, image, detection_result, selected_indices=None):
        """
        Draws pose landmarks on the image with selected indices.

        Args:
            image (np.ndarray): Input image.
            detection_result: Detection result containing landmarks.
            selected_indices (list): List of landmark indices to draw.

        Returns:
            np.ndarray: Annotated image with selected landmarks.
        """
        annotated_image = image.copy()

        # Use selected indices to draw landmarks
        if selected_indices and detection_result.pose_landmarks:
            for index in selected_indices:
                landmark = detection_result.pose_landmarks[0][index]
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                cv2.circle(annotated_image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

        return annotated_image

    def display_image(self, image, title="Pose Landmarks"):
        """
        Displays the given image using matplotlib.

        Args:
            image (np.ndarray): The image to display.
            title (str): Title for the image.
        """
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        plt.axis('off')
        plt.title(title)
        plt.show()