
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
from train import run
from slicing.finalizeDataset import main as finalizeDataset

import glob
import re

import pandas as pd

def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w', encoding="utf-8") as new_file:
        with open(file_path, encoding="utf-8") as old_file:
            for line in old_file:
                new_file.write(re.sub(pattern, subst, line))
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)
  

replace("data/hyps/FlowerHyp.yaml", "fl_gamma: [0-9\.]+", "fl_gamma: 1.5")
replace("data/Flower.yaml", "train: [^\n]+", "train: Reduced")

# Experiment 0:
# Note: remember to set fl_gamma to 1.5 in the yaml file as well as train to the reduced dataset in the other yaml file
# Compare a baseline model with a frozen backbone and a baseline model with a unfrozen backbone

# Downscale dataset to exactly 640 x 640 pixels from 6080 x 3420 pixels, don't slice the images, and use 8 workers
finalizeDataset(
    downscaling_factor="9.5,5.34375",
    num_subset=None,
    verbose="False",
    workers="8",
    slice=False
)
run(
    img=640,
    batch_size=32,
    epochs=8,
    workers=0,
    optimizer="AdamW",
    data="data/Flower.yaml",
    hyp="data/hyps/FlowerHyp.yaml",
    cache="RAM",
    name=f'baseline_model_frozen',
    deterministic=True,
    weights="yolov5s.pt",
    gamma=0,
    class_weights=None,
    freeze=[10]
)
