# Football Video Analysis

A computer-vision pipeline for analyzing football (soccer) video using object detection, multi-object tracking, team assignment, pitch keypoint detection, perspective transformation, and top-down pitch visualization.

The project takes a football video as input, detects and tracks objects such as players, referees, goalkeepers, and the ball, identifies player teams, detects pitch landmarks, and maps detected objects from the camera view to a 2D football-pitch coordinate system.

## Pipeline

```text
Input Video
    |
    v
Player / Ball / Referee Detection
    |
    v
ByteTrack Multi-Object Tracking
    |
    v
Player Team Assignment
    |
    +---------------------+
    |                     |
    v                     v
Pitch Keypoint       Tracked Objects
Detection                  |
    |                       |
    v                       |
Perspective / Homography <-+
Transformation
    |
    v
Top-Down Pitch Visualization
```

## Features

- Football player, goalkeeper, referee, and ball detection using YOLO
- Multi-object tracking using ByteTrack
- Player team assignment using visual/color features and clustering
- Football-pitch keypoint detection
- Mapping from broadcast-camera coordinates to real pitch coordinates
- Top-down 2D pitch visualization
- Annotated output video generation
- Separate notebooks for training the player/object detector and pitch-keypoint detector

## Project Structure

```text
football-analysis/
├── main.py
├── config.py
├── requirements.txt
│
├── tracking/
│   ├── __init__.py
│   └── tracker.py
│
├── team_assignment/
│   ├── __init__.py
│   └── team_assigner.py
│
├── transformation/
│   ├── __init__.py
│   ├── football_coords.py
│   └── transformer.py
│
├── visualization/
│   ├── __init__.py
│   └── p_renderer.py
│
├── utils/
│   ├── __init__.py
│   └── video_utils.py
│
├── notebooks/
│   ├── model_training_players.ipynb
│   └── model_training_field.ipynb
│
├── datasets/             # not tracked by Git
├── models/               # not tracked by Git
└── outputs/              # generated results
```

> The `datasets/` and `models/` directories are intentionally not included in the repository because video datasets and trained model weights can be large.

## Requirements

- Python 3.10+
- PyTorch
- Ultralytics
- Supervision
- OpenCV
- NumPy
- scikit-learn

The exact Python dependencies are listed in `requirements.txt`.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/football-analysis.git
cd football-analysis
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Model Weights

Place the trained model weights inside the `models/` directory:

```text
models/
├── yolo-football-player-detection.pt
└── yolo-football-pitch-detection.pt
```

The project uses two models:

1. **Player/object detector** — detects football-related objects such as players, referees, goalkeepers, and the ball.
2. **Pitch keypoint detector** — detects known football-pitch landmarks used for perspective transformation.

Model weights are not stored directly in this repository because of their size.

<!--
Add download links when the weights are hosted, for example:

- Player detector: <MODEL_DOWNLOAD_URL>
- Pitch detector: <MODEL_DOWNLOAD_URL>
-->

## Dataset

Place input videos inside:

```text
datasets/
```

For example:

```text
datasets/
└── simple.mp4
```

Large datasets and videos should not normally be committed to Git.

## Configuration

Project paths should be defined relative to the repository root in `config.py`.

A typical configuration is:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT / "datasets"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

PLAYER_MODEL_PATH = MODEL_DIR / "yolo-football-player-detection.pt"
PITCH_MODEL_PATH = MODEL_DIR / "yolo-football-pitch-detection.pt"

DEFAULT_VIDEO_PATH = DATASET_DIR / "simple.mp4"
```

Using relative project paths makes the repository portable and avoids machine-specific paths such as `/home/<username>/...`.

## Usage

After placing the required models and video files in their corresponding directories, run:

```bash
python main.py
```

The main processing loop performs the following operations for each frame:

1. Detect football-related objects.
2. Update object tracks with ByteTrack.
3. Extract detected players.
4. Assign players to teams.
5. Detect football-pitch keypoints.
6. Estimate the transformation between image coordinates and pitch coordinates.
7. Transform detected object positions to the top-down pitch.
8. Draw annotations on the original video frame.
9. Draw player, referee, and ball positions on the 2D pitch.
10. Save the generated videos.

## Main Components

### Object Detection and Tracking

`tracking/tracker.py`

The tracker loads the football object-detection model, converts Ultralytics detections into Supervision detections, and uses ByteTrack to preserve object identities across frames.

The tracking stage produces information including:

- bounding boxes
- confidence scores
- object classes
- tracker IDs
- cropped object images

### Team Assignment

`team_assignment/team_assigner.py`

Detected players are separated into teams using visual information extracted from their image crops. The current implementation uses clustering to assign player detections to two team groups.

This allows team membership to be visualized consistently in both the camera view and the top-down pitch representation.

### Pitch Detection

`transformation/transformer.py`

A keypoint-detection YOLO model detects known landmarks on the football pitch.

Only sufficiently confident keypoints are retained and used for the perspective transformation.

### Pitch Coordinate System

`transformation/football_coords.py`

Defines the football-pitch geometry and landmark coordinates used as reference points for transforming image coordinates into pitch coordinates.

### Visualization

`visualization/p_renderer.py`

Handles visualization of detected objects on a top-down football pitch, including:

- players
- referees
- goalkeeper positions
- ball position
- pitch geometry

### Video Utilities

`utils/video_utils.py`

Contains helper functions for reading and saving video data.

## Model Training

Training notebooks are included for the two major vision models.

### Player/Object Detection

```text
notebooks/model_training_players.ipynb
```

Used to train and evaluate the football player/object detection model.

### Pitch Keypoint Detection

```text
notebooks/model_training_field.ipynb
```

Used to train and evaluate the football-pitch landmark/keypoint detector.

Training runs, checkpoints, and large generated artifacts should normally be excluded from Git and stored separately.

## Output

The pipeline produces two main forms of output:

### Annotated Camera View

The original football video with tracked objects, object IDs, detected pitch keypoints, and class/team annotations.

### Top-Down Pitch View

A 2D representation of the football pitch showing transformed player, referee, goalkeeper, and ball positions.

<!--
For a public GitHub repository, add a few example images or GIFs here:

![Detection Example](assets/detection_example.jpg)

![Top Down View](assets/top_down_example.jpg)
-->

## Recommended `.gitignore`

Large files, model weights, datasets, generated runs, and Python cache files should not be committed.

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/

# Datasets and models
datasets/
models/

# Generated outputs
outputs/
runs/
runs_field/

# Video files
*.mp4
*.avi
*.mov
*.mkv

# OS files
.DS_Store
Thumbs.db
```

If you want to keep selected result images for the README, copy only those examples into an `assets/` directory.

## Current Limitations

- Team assignment is primarily appearance/color based and can become less reliable under strong illumination changes or visually similar kits.
- Pitch transformation depends on detecting enough reliable pitch keypoints.
- Occlusion, motion blur, and difficult camera angles can reduce detection and tracking quality.
- The current pipeline is designed around broadcast-style football footage and may require adaptation for other camera setups.

## Possible Improvements

Future improvements could include:

- more robust team classification across an entire match
- player identity re-identification
- trajectory and heatmap visualization
- player speed and distance estimation
- possession estimation
- automatic event detection
- improved ball tracking during occlusion
- tactical formation analysis
- camera-motion-aware tracking
- exporting tracking and pitch-coordinate data to CSV or JSON

## Technologies

- Python
- PyTorch
- Ultralytics YOLO
- Supervision
- ByteTrack
- OpenCV
- NumPy
- scikit-learn
- Jupyter Notebook

## Repository Notes

Model checkpoints, datasets, input videos, and complete training-run directories are intentionally excluded from source control because of their size.

For reproducibility, the repository should contain:

- source code
- training notebooks
- dependency definitions
- configuration examples
- instructions for obtaining model weights and datasets
- a small number of representative result images

while large generated artifacts should be hosted separately.

## License

Add a license before publishing the repository if you want to clearly define how other people may use the code.

For example, the MIT License is commonly used for open-source software projects.

---

If you use or extend this project, contributions and suggestions are welcome.
