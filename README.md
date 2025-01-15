# Get Fit With PCA : Video Analysis Framework

![Fitness App](images/banner_image.png)

### Abstract

In this project, we developed a modular framework to evaluate the correctness of exercise videos (dumbbell shoulder press exercise). We utilized MedidaPipe library for extracting the landmarks and a statistical modeling technique called Principal Component Analysis (PCA) for our analysis.

### Features
- Extract frames from videos at specified intervals.
- Process frames to extract pose landmarks and then normalize them using Rigid alignment.
- Generate and evaluate parameters using PCA.
- Provide a correctness score for test videos.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mohit-choithwani/Get_Fit_with_PCA
cd Get_Fit_with_PCA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Prepare Your Videos - Place training and test videos in separate directories.
Update paths and parameters in the main.py script and execute:
```bash
python main.py
```

### How It Works

1. **Video Input:** The user records a video of themselves performing the exercise.
2. **Landmark Detection:** The application processes each frame of the video using the MediaPipe library to extract 23 relevant body landmarks, which are stored in a CSV file.
3. **Data Preprocessing:** Rigid alignment is applied to all landmarks to ensure consistent positioning, and mean normalization is used to avoid outliers.
4. **Principal Component Analysis:** The mean-normalized data is used for PCA, calculating parameters such as Eigenvectors and Eigenvalues.
5. **Correctness Evaluation (L2 Norm Scoring):** Each frame is analyzed to determine if the exercise is performed correctly.
6. **Scoring:** Based on the calculated L2 norm and predefined threshold. The model provides a numerical score indicating the accuracy of the exercise performance.

![Incorrect_exercise](images/pca_image.gif)


### Future Work

This model offers a low-data solution to exercise assessment, aimed at helping users perform exercises properly and effectively without needing a coach. Future enhancements include supporting additional exercises, non-frontal video processing, real-time performance analysis, and mobile application development.

