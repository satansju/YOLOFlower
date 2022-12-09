
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
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
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

# Experiment 1: Compare the baseline model and the sliced model with a downscaling factor of 2, 3, 4, and 5
# Note: Remember to set fl_gamma to 1.5 in the yaml file
for D, A in zip([2.5, 3.5, 4.5], [20, 15, 10]):
    finalizeDataset(
        downscaling_factor=str(D), 
        num_subset=None, 
        verbose="False", 
        workers="8"
    )
    D = str(D).replace(".", "dot")
    run(
        img=640,
        batch_size=32,
        epochs=8,
        workers=0,
        optimizer="AdamW",
        data="data/Flower.yaml",
        hyp="data/hyps/FlowerHyp.yaml",
        cache="RAM",
        name=f'sliced_model_D{D}_equal',
        deterministic=True,
        weights="yolov5s.pt",
        gamma=0,
        class_weights=None,
        dataset_size=2000,
        anchors=A
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
        name=f'sliced_model_D{D}_full',
        deterministic=True,
        weights="yolov5s.pt",
        gamma=0,
        class_weights=None,
        anchors=A
    )
