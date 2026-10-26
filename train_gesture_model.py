
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "hand_gesture_dataset" / "landmark_samples.csv"
MODEL_DIR = BASE_DIR / "models"

CLASSIFIER_FILE = MODEL_DIR / "gesture_classifier.pkl"
SCALER_FILE = MODEL_DIR / "gesture_scaler.pkl"
METADATA_FILE = MODEL_DIR / "model_metadata.json"

FEATURE_COLUMNS = [
    f"{axis}{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]


class DatasetNeedsSamplesError(ValueError):
    """Raised when the dataset has no valid samples to train on."""


def load_dataset():
    if not DATASET_FILE.exists():
        raise DatasetNeedsSamplesError(
            f"No landmark dataset found at:\n{DATASET_FILE}"
        )

    features = []
    labels = []

    with DATASET_FILE.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        required = {"gesture", *FEATURE_COLUMNS}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                "Dataset columns are missing: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            try:
                label = row["gesture"].strip()
                values = [float(row[col]) for col in FEATURE_COLUMNS]

                if not label or not np.isfinite(values).all():
                    continue

                features.append(values)
                labels.append(label)

            except (ValueError, TypeError):
                continue

    if not features:
        raise DatasetNeedsSamplesError(
            f"No valid landmark samples found in:\n{DATASET_FILE}\n\n"
            "The dataset is empty or contains no valid landmark rows."
        )

    return np.asarray(features, dtype=np.float32), np.asarray(labels)


def load_or_collect_dataset():
    try:
        return load_dataset()
    except DatasetNeedsSamplesError:
        print(
            "\nNo training samples are available. Starting the hand "
            "tracker now."
        )
        print(
            "Focus the webcam window and press S to save samples. "
            "Collect at least two samples for each of at least two "
            "gestures, then press Q to continue training.\n"
        )
        subprocess.run(
            [sys.executable, str(BASE_DIR / "hand_gesture_recognition.py")],
            check=True,
        )
        return load_dataset()


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_or_collect_dataset()
    classes, counts = np.unique(y, return_counts=True)

    print("\nGESTURE MODEL TRAINING")
    print("----------------------")
    print(f"Total samples: {len(X)}")
    print(f"Features per sample: {X.shape[1]}")

    for name, count in zip(classes, counts):
        print(f"{name}: {count} samples")

    if len(classes) < 2:
        raise ValueError(
            "Collect samples for at least TWO different gestures."
        )

    if counts.min() < 2:
        raise ValueError(
            "Each gesture needs at least TWO samples. "
            "Collect more samples and try again."
        )

    # Use a test set only when every class can be represented in it.
    test_count = max(len(classes), round(len(X) * 0.2))
    can_split = (
        len(X) - test_count >= len(classes)
        and test_count < len(X)
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=3.0,
            gamma="scale",
            class_weight="balanced",
        )),
    ])

    if can_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_count,
            random_state=42,
            stratify=y,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        print(f"\nTest accuracy: {accuracy:.2%}")
        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            zero_division=0,
        ))
    else:
        print(
            "\nDataset too small for a reliable stratified test split."
        )
        print("Training on all available samples; accuracy not evaluated.")
        model.fit(X, y)
        accuracy = None

    # Save the fitted classifier and scaler separately.
    joblib.dump(model.named_steps["classifier"], CLASSIFIER_FILE)
    joblib.dump(model.named_steps["scaler"], SCALER_FILE)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": "Support Vector Machine (RBF)",
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "classes": classes.tolist(),
        "samples_per_class": {
            name: int(count)
            for name, count in zip(classes, counts)
        },
        "total_samples": int(len(X)),
        "test_accuracy": accuracy,
        "coordinate_processing": (
            "Coordinates relative to wrist, normalized "
            "by maximum absolute coordinate"
        ),
    }

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("\nTraining completed.")
    print(f"Classifier: {CLASSIFIER_FILE}")
    print(f"Scaler:     {SCALER_FILE}")
    print(f"Metadata:   {METADATA_FILE}")
    print(
        "\nNote: test accuracy is an estimate, "
        "not a guarantee of real-world performance."
    )


if __name__ == "__main__":
    main()
