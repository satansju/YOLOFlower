from train import run
from slicing.finalizeDataset import main as finalizeDataset

import glob
import re

import pandas as pd

from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove

# This file contains the code to run the experiments.
# The experiments are run by uncommenting the code below. Only uncomment the code for the experiment(s) you want to run.

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
  
# Baseline:    
# replace("data/Flower.yaml", "train: [^\n]+", "train: Reduced")

# Sliced:
# replace("data/Flower.yaml", "train: [^\n]+", "train: Sliced")

# With focal loss:
# replace("data/hyps/FlowerHyp.yaml", "fl_gamma: [0-9\.]+", "fl_gamma: 1.5")

# Without focal loss:
# replace("data/hyps/FlowerHyp.yaml", "fl_gamma: [0-9\.]+", "fl_gamma: 0")
  

# Example of how to run the code with a downscaling factor of 2, gamma of 0, an equal dataset, 15 anchors and 8 epochs.
# finalizeDataset(
#     downscaling_factor="2", 
#     num_subset=None, 
#     verbose="False", 
#     workers="8"
# )
# 
# run(
#     img=640,
#     batch_size=32,
#     epochs=8,
#     workers=0,
#     optimizer="AdamW",
#     data="data/Flower.yaml",
#     hyp="data/hyps/FlowerHyp.yaml",
#     cache="RAM",
#     name=f'test_anchors',
#     deterministic=True,
#     weights="yolov5s.pt",
#     gamma=0,
#     class_weights=None,
#     dataset_size=2000,
#     anchors=15
# )
#
# End of example

# Experiment 0:
# Compare a baseline model with a frozen backbone and a baseline model with a unfrozen backbone

# replace("data/hyps/FlowerHyp.yaml", "fl_gamma: [0-9\.]+", "fl_gamma: 1.5")
# replace("data/Flower.yaml", "train: [^\n]+", "train: Reduced")
# Downscale dataset to exactly 640 x 640 pixels from 6080 x 3420 pixels, don't slice the images, and use 8 workers
# finalizeDataset(
#     downscaling_factor="9.5,5.34375",
#     num_subset=None,
#     verbose="False",
#     workers="8",
#     slice=False
# )
# run(
#     img=640,
#     batch_size=32,
#     epochs=8,
#     workers=0,
#     optimizer="AdamW",
#     data="data/Flower.yaml",
#     hyp="data/hyps/FlowerHyp.yaml",
#     cache="RAM",
#     name=f'baseline_model_frozen',
#     deterministic=True,
#     weights="yolov5s.pt",
#     gamma=0,
#     class_weights=None,
#     freeze=[10]
# )
# run(
#     img=640,
#     batch_size=32,
#     epochs=8,
#     workers=0,
#     optimizer="AdamW",
#     data="data/Flower.yaml",
#     hyp="data/hyps/FlowerHyp.yaml",
#     cache="RAM",
#     name=f'baseline_model',
#     deterministic=True,
#     weights="yolov5s.pt",
#     gamma=0,
#     class_weights=None
# )
# End of experiment 0

# Experiment 1: 
# Compare the baseline model and the sliced model with a downscaling factor of 2, 3, 4, and 5

# replace("data/hyps/FlowerHyp.yaml", "fl_gamma: [0-9\.]+", "fl_gamma: 1.5")
# replace("data/Flower.yaml", "train: [^\n]+", "train: Sliced")
# for D, A in zip([2.5, 3.5, 4.5], [20, 15, 10]):
#     finalizeDataset(
#         downscaling_factor=str(D), 
#         num_subset=None, 
#         verbose="False", 
#         workers="8"
#     )
#     D = str(D).replace(".", "dot")
#     run(
#         img=640,
#         batch_size=32,
#         epochs=8,
#         workers=0,
#         optimizer="AdamW",
#         data="data/Flower.yaml",
#         hyp="data/hyps/FlowerHyp.yaml",
#         cache="RAM",
#         name=f'sliced_model_D{D}_equal',
#         deterministic=True,
#         weights="yolov5s.pt",
#         gamma=0,
#         class_weights=None,
#         dataset_size=2000,
#         anchors=A
#     )
#     run(
#         img=640,
#         batch_size=32,
#         epochs=8,
#         workers=0,
#         optimizer="AdamW",
#         data="data/Flower.yaml",
#         hyp="data/hyps/FlowerHyp.yaml",
#         cache="RAM",
#         name=f'sliced_model_D{D}_full',
#         deterministic=True,
#         weights="yolov5s.pt",
#         gamma=0,
#         class_weights=None,
#         anchors=A
#     )
# End of experiment 1

# Experiment 2: Compare the baseline model and the sliced model with a gamma of 0, 0.5, 1, 1.5, 2 and class weights on and off
# replace("data/hyps/FlowerHyp.yaml", "fl_gamma: [0-9\.]+", "fl_gamma: 0")
# replace("data/Flower.yaml", "train: [^\n]+", "train: Sliced")
#
# finalizeDataset(
#     downscaling_factor="4",
#     num_subset=None,
#     verbose="False",
#     workers="8"
# )
#
# for gamma in [0, 1, 2]:
#     for class_weights in [False, True]:
#         clw_lab = 'Y' if class_weights else 'N'
#         gamma_lab = re.sub("\.", "dot", str(gamma))
#         run(
#             img=640,
#             batch_size=32,
#             epochs=25,
#             workers=8,
#             optimizer="AdamW",
#             data="data/Flower.yaml",
#             hyp="data/hyps/FlowerHyp.yaml",
#             cache="RAM",
#             name=f'sliced_model_D4_full_gamma{gamma_lab}_class_weights{clw_lab}',
#             deterministic=True,
#             weights="yolov5s.pt",
#             gamma=gamma,
#             class_weights=class_weights
#         )
# End of experiment 2

# Final results:
## Use the best best values of gamma and class weights for training over 100 epochs on the full dataset (downscaling factor 4)
# replace("data/hyps/FlowerHyp.yaml", "fl_gamma: [0-9\.]+", "fl_gamma: 0")  
# replace("data/Flower.yaml", "train: [^\n]+", "train: Sliced")

# Slice dataset
# finalizeDataset(
#     downscaling_factor="4",
#     num_subset=None,
#     verbose="False",
#     workers="8"
# )
# Train final model for 200 epochs
# run(
#     name= "final_model_200_epochs",
#     img=640,
#     batch_size=32,
#     epochs=200,
#     workers=8,
#     optimizer="AdamW", # If the training is not stable (i.e. loss becomes nan), try SGD
#     data="data/Flower.yaml",
#     hyp="data/hyps/FlowerHyp.yaml",
#     cache="RAM",
#     deterministic=True,
#     weights="yolov5s.pt",
#     gamma=0,
#     class_weights=True
# )

# Train baseline model for 200 epochs
# replace("data/hyps/FlowerHyp.yaml", "fl_gamma: [0-9\.]+", "fl_gamma: 1.5")
# replace("data/Flower.yaml", "train: [^\n]+", "train: Reduced")
# 
# Create reduced, but not sliced, dataset
# finalizeDataset(
#     downscaling_factor="1",
#     num_subset=None,
#     verbose="False",
#     workers="8",
#     slice=False
# )
# Train baseline model for 200 epochs
# run(
#     name= "baseline_model_focal_200_epochs",
#     img=640,
#     batch_size=32,
#     epochs=200,
#     workers=8,
#     optimizer="AdamW",
#     data="data/Flower.yaml",
#     hyp="data/hyps/FlowerHyp.yaml",
#     cache="RAM",
#     deterministic=True,
#     weights="yolov5s.pt",
#     gamma=0,
#     class_weights=False
# )
# Disable focal loss (set fl_gamma to 0)
# replace("data/hyps/FlowerHyp.yaml", "fl_gamma: [0-9\.]+", "fl_gamma: 0")
# Train baseline model for 200 epochs
# run(
#     name= "baseline_model_NOfocal_200_epochs",
#     img=640,
#     batch_size=32,
#     epochs=200,
#     workers=8,
#     optimizer="AdamW",
#     data="data/Flower.yaml",
#     hyp="data/hyps/FlowerHyp.yaml",
#     cache="RAM",
#     deterministic=True,
#     weights="yolov5s.pt",
#     gamma=0,
#     class_weights=False
# )
# End of final results
# The code below is used to prepare the final model for further use (uncomment if needed)

# Prepare the setups for further use of the final model:
# 1. Disable focal loss (set fl_gamma to 0)
# replace("data/hyps/FlowerHyp.yaml", "fl_gamma: [0-9\.]+", "fl_gamma: 0")
# 2. Use the sliced dataset
# replace("data/Flower.yaml", "train: [^\n]+", "train: Sliced")
# 3. Slice dataset (downscaling factor 4)
# finalizeDataset(
#     downscaling_factor="4",
#     num_subset=None,
#     verbose="False",
#     workers="8"
# )


# Auxillary runs
# 1) Without the "Withered" class
finalizeDataset(
    downscaling_factor="4",
    num_subset=None,
    verbose="False",
    workers="8",
    slice=False,
    exclude_classes=["Withered", "Gone"]
)
run(
    name= "final_model_200_epochs_NoWithered",
    img=640,
    batch_size=32,
    epochs=200,
    workers=8,
    optimizer="AdamW", # If the training is not stable (i.e. loss becomes nan), try SGD
    data="data/Flower.yaml",
    hyp="data/hyps/FlowerHyp.yaml",
    cache="RAM",
    deterministic=True,
    weights="yolov5s.pt",
    gamma=0,
    class_weights=True
)