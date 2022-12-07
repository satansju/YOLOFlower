from train import run
from slicing.finalizeDataset import main as finalizeDataset

import glob
import re

import pandas as pd

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


# Experiment 1: Compare the baseline model and the sliced model with a downscaling factor of 2, 3, 4, and 5
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

# Experiment 2: Compare the baseline model and the sliced model with a gamma of 0, 0.5, 1, 1.5, 2 and class weights on and off

# Test different values of gamma and class weights for downscaling factor 4 (full dataset)
# finalizeDataset(
#     downscaling_factor="4",
#     num_subset=None,
#     verbose="False",
#     workers="8"
# )

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

# # Use the best best values of gamma and class weights for training over 100 epochs on the full dataset (downscaling factor 4)

# finalizeDataset(
#     downscaling_factor="4",
#     num_subset=None,
#     verbose="False",
#     workers="8"
# )

# A test run without focal loss, with class weights and gamma of 0

# run(
#     img=640,
#     batch_size=32,
#     epochs=50,
#     workers=8,
#     optimizer="AdamW",
#     data="data/Flower.yaml",
#     hyp="data/hyps/FlowerHyp.yaml",
#     cache="RAM",
#     name=f'test_model_D4_full_gamma0_class_weightsY_focal_lossY_epoch50',
#     deterministic=True,
#     weights="yolov5s.pt",
#     gamma=0,
#     class_weights=True
# )

# run(
#     img=640,
#     batch_size=32,
#     epochs=50,
#     workers=8,
#     optimizer="AdamW",
#     data="data/Flower.yaml",
#     hyp="data/hyps/FlowerHyp.yaml",
#     cache="RAM",
#     name=f'test_model_D4_full_gamma0_class_weightsY_focal_lossY_epoch50',
#     deterministic=True,
#     weights="yolov5s.pt",
#     gamma=1,
#     class_weights=True
# )


# Find the best values of gamma and class weights by looking at the all of the results of the previous experiments in the runs/train folder

previous_results = glob.glob("runs/train/sliced_model_D4_full_gamma*class_weights*")

# Results are stored in a tabular separated file called results_class.csv in each of the previous results folders
# An example of the content of this file is shown below
# Class	Images	Instances	P	R	mAP50	mAP50-95
# all	201	12698	0.678324621165275	0.16687968179506804	0.09634375754331848	0.03490617805157518
# Bud	201	1259	1.0	0.0	0.011588884792626725	0.003250673135578406
# Flower	201	1719	0.24523136574727764	0.4653868528214078	0.35129944359577014	0.13597552924769313
# Withered	201	8913	0.14639174007909764	0.36901155615393244	0.1188304593281955	0.03530468787460432
# Immature	201	499	1.0	0.0	0.0	0.0
# Mature	201	308	1.0	0.0	0.0	0.0

columns = ["Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95", "gamma", "class_weights"]

results = pd.DataFrame(columns=columns)

for result in previous_results:
    result = result + "/class_results.csv"
    g = re.findall("gamma(.*)_class_weights", result)[0]
    g = float(re.sub("dot", ".", g))
    cw = re.findall("class_weights(.*)", result)[0]
    cw = True if cw == "Y" else False
    result = pd.read_csv(result, sep="\t")
    result["gamma"] = g
    result["class_weights"] = cw
    results = results.append(result)
    
results = results.reset_index(drop=True)

# Get the best values of gamma and class weights

best_gamma = results.groupby("gamma").mean().sort_values("mAP50-95", ascending=False).index[0]
best_class_weights = results.groupby("class_weights").mean().sort_values("mAP50-95", ascending=False).index[0]

# Test the code above by running the following command

print(f"Best gamma: {best_gamma}, best class weights: {best_class_weights}")

assert False, "Not ready to run the full training"

# Train the model with the best values of gamma and class weights for 100 epochs

run(
    img=640,
    batch_size=32,
    epochs=100,
    workers=0,
    optimizer="AdamW",
    data="data/Flower.yaml",
    hyp="data/hyps/FlowerHyp.yaml",
    cache="RAM",
    name=f'sliced_model_D4dot5_full_gamma{best_gamma}_class_weights{best_class_weights}_100epochs',
    deterministic=True,
    weights="yolov5s.pt",
    gamma=float(best_gamma),
    class_weights=best_class_weights == "Y"
)

