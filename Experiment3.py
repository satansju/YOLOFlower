
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
  

replace("data/hyps/FlowerHyp.yaml", "fl_gamma: [0-9\.]+", "fl_gamma: 0")  
replace("data/Flower.yaml", "train: [^\n]+", "train: Sliced")

finalizeDataset(
    downscaling_factor="4",
    num_subset=None,
    verbose="False",
    workers="8"
)

run(
    name= "final_model_200_epochs",
    img=640,
    batch_size=32,
    epochs=200,
    workers=8,
    optimizer="AdamW",
    data="data/Flower.yaml",
    hyp="data/hyps/FlowerHyp.yaml",
    cache="RAM",
    deterministic=True,
    weights="yolov5s.pt",
    gamma=0,
    class_weights=True
)

replace("data/hyps/FlowerHyp.yaml", "fl_gamma: [0-9\.]+", "fl_gamma: 1.5")
replace("data/Flower.yaml", "train: [^\n]+", "train: Reduced")

finalizeDataset(
    downscaling_factor="1",
    num_subset=None,
    verbose="False",
    workers="8",
    slice=False
)

run(
    name= "baseline_model_focal_200_epochs",
    img=640,
    batch_size=32,
    epochs=200,
    workers=8,
    optimizer="AdamW",
    data="data/Flower.yaml",
    hyp="data/hyps/FlowerHyp.yaml",
    cache="RAM",
    deterministic=True,
    weights="yolov5s.pt",
    gamma=0,
    class_weights=False
)

replace("data/hyps/FlowerHyp.yaml", "fl_gamma: [0-9\.]+", "fl_gamma: 0")
run(
    name= "baseline_model_NOfocal_200_epochs",
    img=640,
    batch_size=32,
    epochs=200,
    workers=8,
    optimizer="AdamW",
    data="data/Flower.yaml",
    hyp="data/hyps/FlowerHyp.yaml",
    cache="RAM",
    deterministic=True,
    weights="yolov5s.pt",
    gamma=0,
    class_weights=False
)

replace("data/hyps/FlowerHyp.yaml", "fl_gamma: [0-9\.]+", "fl_gamma: 0")

finalizeDataset(
    downscaling_factor="4",
    num_subset=None,
    verbose="False",
    workers="8"
)
