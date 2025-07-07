import fiftyone as fo
import fiftyone.zoo as foz
import os
from fiftyone.utils.openimages import get_classes
from random import sample

# we need to run this dataset download using a main guard 
# as it ensures code only runs when directly called as the main script and
# not when the script is imported as a module or reloaded in a child process.
# that can happen through the fiftyone module which spins up multiprocessing 
# and causes things to grind to a halt...
# ultimately is because windows uses the spawn method to re import a script fully 
# versus linux which uses a fork method (majority of users are linux)
# upon triggering a new process via multiprocessing, subprocesses, etc..

# Get all boxable classes
all_classes = get_classes()

# Filter out unwanted classes
excluded = {"Cat", "Dog", "Carnivore", "Mammal", "Animal"}
safe_classes = [c for c in all_classes if c not in excluded]

# Choose 150 random safe classes
subset = sample(safe_classes, 150)

# Load dataset with strict filtering
dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=subset,
        max_samples=150,
        shuffle=True
    )


if __name__ == "__main__":
    
    # Start FiftyOne App 
    session = fo.launch_app(dataset)
    session.wait()


    # # Export dataset to a folder named 'non_catdog_images'
    # dataset.export(
    #     export_dir="non_catdog_images",
    #     dataset_type=foz.types.ImageClassificationDirectoryTree,
    #     label_field="ground_truth"
    # )