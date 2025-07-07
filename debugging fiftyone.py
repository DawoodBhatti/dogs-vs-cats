import os
import logging

# Set these BEFORE importing fiftyone
os.environ["FIFTYONE_DO_NOT_TRACK"] = "true"
os.environ["FIFTYONE_DISABLE_NOTEBOOK"] = "1"

import fiftyone as fo
import fiftyone.core.logging as fol

fo.config.database_uri = "mongodb://localhost:27017"
fol.set_logging_level(logging.DEBUG)

if __name__ == "__main__":
    dataset = fo.zoo.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections", "segmentations", "points"],
        classes=["Cat", "Dog"],
        max_samples=100,
    )

    session = fo.launch_app(dataset, port=5152)
    session.wait()
