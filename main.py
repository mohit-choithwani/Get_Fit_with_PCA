import os
import pickle
from pathlib import Path
from video_processing import process_video_folder
from frame_processing import process_frames_and_extract_landmarks

def main():
    # Define video paths and frame paths
    TEST_VIDEO_PATH = r"D:\Get_Fit_with_PCA\test_videos\correct_videos"
    TEST_OUTPUT_DIR = r"D:\Get_Fit_with_PCA\test_corr_frames"
    MODEL_PATH = r"pose_landmarker_heavy.task"
    FRAME_INTERVAL = 10
    OUTPUT_PICKLE_FILE = "test_incorrect_extracted_landmarks.pkl"
    
    # Ensure output directories exist
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Process videos and extract frames
    print(f"Processing videos in {TEST_VIDEO_PATH}...")
    process_video_folder(TEST_VIDEO_PATH, TEST_OUTPUT_DIR, FRAME_INTERVAL)
    print(f"Frames saved to {TEST_OUTPUT_DIR}")
    
    # Step 2: Process frames and extract landmarks
    print(f"Extracting landmarks from frames in {TEST_OUTPUT_DIR}...")
    test_extracted_landmarks = process_frames_and_extract_landmarks(TEST_OUTPUT_DIR, MODEL_PATH)
    print("Landmark extraction complete.")
    
    # Step 3: Save the intermediate results
    with open(OUTPUT_PICKLE_FILE, 'wb') as f:
        pickle.dump(test_extracted_landmarks, f)
    print(f"Landmarks saved to {OUTPUT_PICKLE_FILE}")

if __name__ == "__main__":
    main()
