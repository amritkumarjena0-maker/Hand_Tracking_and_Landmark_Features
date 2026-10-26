# Hand Tracking and Landmark Features

## Description
A Python-based Advanced Computer Vision project that uses MediaPipe, OpenCV, and Machine Learning to detect hands, extract 21 hand landmarks, and classify hand gestures using landmark features.

The project captures hand movements through a webcam, stores landmark samples in a CSV dataset, and trains a machine learning classifier using normalized landmark coordinates.

## Features
- Real-time webcam hand tracking.
- Detection of 21 hand landmarks.
- Extraction of 63 landmark features (x, y, z coordinates).
- Handedness detection.
- Gesture sample collection and CSV dataset generation.
- Machine learning model training using scikit-learn.
- Feature scaling and Support Vector Machine (SVM) classification.
- Model evaluation using a classification report.
- CSV tracking logs.
- Saved classifier, scaler, and model metadata.

## Technologies
- Python
- OpenCV
- MediaPipe
- NumPy
- scikit-learn
- Joblib
- CSV
- JSON

## Suggested Project Folder Structure

```text
Hand_Tracking_and_Landmark_Features/
├── hand_gesture_recognition.py
├── train_gesture_model.py
├── hand_gesture_dataset/
│   └── landmark_samples.csv
├── models/
│   ├── gesture_classifier.pkl
│   ├── gesture_scaler.pkl
│   └── model_metadata.json
├── output/
│   └── tracking_log.csv
└── README.md
```

## Gesture Classes
The current dataset contains five gesture labels:

- CUSTOM
- OPEN PALM
- POINTING
- FIST
- PEACE

## Algorithm

1. Start the webcam using OpenCV.
2. Capture each video frame.
3. Process the frame using MediaPipe Hands.
4. Detect the hand and extract its 21 landmarks.
5. Normalize landmark coordinates relative to the wrist.
6. Convert the 21 landmarks into 63 numerical features.
7. Display hand landmarks and gesture information.
8. Save landmark samples and tracking logs when requested.
9. Load the collected CSV dataset in the training program.
10. Scale the features and train an SVM classifier.
11. Evaluate the model using a test dataset.
12. Save the trained classifier, scaler, and metadata.

## Flowchart

```text
          START
            |
            v
      Open Webcam
            |
            v
      Capture Frame
            |
            v
    Detect Hand Using
        MediaPipe
            |
            v
    Extract 21 Landmarks
            |
            v
    Normalize Coordinates
            |
            v
    Generate 63 Features
            |
            v
    Display Hand Tracking
            |
            v
    Save Samples to CSV
            |
            v
       Train SVM Model
            |
            v
     Evaluate Classifier
            |
            v
       Save Model Files
            |
            v
           END
```

## How to Run

Run the webcam application:

```powershell
.\.venv\Scripts\python.exe hand_gesture_recognition.py
```

Run the model training program:

```powershell
.\.venv\Scripts\python.exe train_gesture_model.py
```

### Webcam Controls

- **S** — Save the current detected hand sample.
- **Q** — Quit the webcam application.

## Training Results

The model has been trained successfully using the collected landmark dataset.

| Metric | Result |
|---|---:|
| Total samples | 39 |
| Gesture classes | 5 |
| Features per sample | 63 |
| Test samples | 8 |
| Test accuracy | 87.50% |

The reported test accuracy corresponds to 7 correctly classified samples out of 8. Because the test set is small and the dataset is unevenly distributed across gesture classes, this result should be treated as an initial estimate rather than a guarantee of real-world accuracy.

## Generated Model Files

- `models/gesture_classifier.pkl` — Trained gesture classifier.
- `models/gesture_scaler.pkl` — Feature scaling model.
- `models/model_metadata.json` — Model and feature metadata.

## Output Files

- `hand_gesture_dataset/landmark_samples.csv` — Saved landmark samples and gesture labels.
- `output/tracking_log.csv` — Hand-tracking experiment logs.
- `models/` — Trained model artifacts.

## Note

The current project supports landmark collection and model training. For machine-learning-based live predictions, the webcam application must load and use the trained classifier and scaler. Recognition quality depends on dataset size, label quality, gesture variation, and testing conditions.

## Author

Amrit 
