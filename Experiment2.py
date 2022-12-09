
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
  

replace("data/hyps/FlowerHyp.yaml", "fl_gamma: [0-9\.]+", "fl_gamma: 0")
replace("data/Flower.yaml", "train: [^\n]+", "train: Reduced")

# Experiment 2: Compare the baseline model and the sliced model with a gamma of 0, 0.5, 1, 1.5, 2 and class weights on and off
# Note: Remember to set fl_gamma to 0 in the yaml file

finalizeDataset(
    downscaling_factor="4",
    num_subset=None,
    verbose="False",
    workers="8"
)

for gamma in [0, 1, 2]:
    for class_weights in [False, True]:
        clw_lab = 'Y' if class_weights else 'N'
        gamma_lab = re.sub("\.", "dot", str(gamma))
        run(
            img=640,
            batch_size=32,
            epochs=25,
            workers=8,
            optimizer="AdamW",
            data="data/Flower.yaml",
            hyp="data/hyps/FlowerHyp.yaml",
            cache="RAM",
            name=f'sliced_model_D4_full_gamma{gamma_lab}_class_weights{clw_lab}',
            deterministic=True,
            weights="yolov5s.pt",
            gamma=gamma,
            class_weights=class_weights
        )

## Use the best best values of gamma and class weights for training over 100 epochs on the full dataset (downscaling factor 4)
