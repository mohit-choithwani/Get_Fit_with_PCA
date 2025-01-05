import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import mediapipe as mp

def plot_landmarks_on_image(image_path, landmarks_array):
    """
    Draws landmarks on the given image based on the provided NumPy array.

    Args:
        image_path (str): Path to the input image.
        landmarks_array (np.ndarray): A NumPy array of shape (N, 2), where N is the number of landmarks.
                                       Each row contains (x, y) normalized coordinates.
        connections (list of tuples): Optional; List of landmark connections as pairs of indices (e.g., [(0, 1), (1, 2)]).
    
    Returns:
        None: Displays the annotated image with landmarks and connections.
    """
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Check if the image is loaded properly
    if image is None:
        raise ValueError(f"Could not load image from path: {image_path}")

    # Convert the image to RGB format (OpenCV loads in BGR by default)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # get x and y cordinates
    x_points, y_points = landmarks_array[:,0], landmarks_array[:,1]

    # Get image dimensions
    width, height, _ = rgb_image.shape

    # Convert normalized coordinates to pixel coordinates
    new_x = [x*height for x in x_points]
    new_y = [y*width for y in y_points]
    
    plt.imshow(image)
    plt.scatter(x=new_x, y=new_y, c='r', s=5)
    plt.axis("off")
    plt.show()

